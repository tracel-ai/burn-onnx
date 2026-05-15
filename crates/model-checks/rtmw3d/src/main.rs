extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

// Import the generated model code as a module
pub mod rtmw3d {
    include!(concat!(env!("OUT_DIR"), "/model/rtmw3d_opset16.rs"));
}

#[derive(Debug, Module)]
struct TestData {
    input: Param<Tensor<4>>,
    output: Param<Tensor<3>>,
}

impl TestData {
    fn new(device: &Device) -> Self {
        // RTMW3D-x: input [1, 3, 384, 288], primary output [1, 133, 576]
        Self {
            input: Initializer::Zeros.init([1, 3, 384, 288], device),
            output: Initializer::Zeros.init([1, 133, 576], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("RTMW3D-x Burn Model Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("rtmw3d");
    println!("Artifacts directory: {}", artifacts_dir.display());

    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    // Initialize the model
    println!("Initializing RTMW3D-x model...");
    let start = Instant::now();
    let device = model_checks_common::best_device!();
    let weights_path = concat!(env!("OUT_DIR"), "/model/rtmw3d_opset16.bpk");
    let model: rtmw3d::Model = rtmw3d::Model::from_file(weights_path, &device);
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Save model structure to file
    let model_txt_path = artifacts_dir.join("model.txt");
    println!(
        "\nSaving model structure to {}...",
        model_txt_path.display()
    );
    let model_str = format!("{}", model);
    std::fs::write(&model_txt_path, &model_str).expect("Failed to write model structure to file");
    println!("  Model structure saved");

    // Load test data from PyTorch file
    let test_data_path = artifacts_dir.join("test_data.pt");
    println!("\nLoading test data from {}...", test_data_path.display());
    let start = Instant::now();
    let mut test_data = TestData::new(&device);
    let mut store = PytorchStore::from_file(&test_data_path);
    test_data
        .load_from(&mut store)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    // Get the input tensor
    let input = test_data.input.val();
    let input_shape: [usize; 4] = input.shape().dims();
    println!("  Loaded input with shape: {:?}", input_shape);

    // Get the reference output (primary output only)
    let reference_output = test_data.output.val();
    let ref_shape: [usize; 3] = reference_output.shape().dims();
    println!("  Loaded reference output with shape: {:?}", ref_shape);

    // Warmup run (compiles GPU shaders, allocates buffers)
    println!("\nWarmup inference...");
    let start = Instant::now();
    let _ = model.forward(input.clone());
    println!("  Warmup completed in {:.2?}", start.elapsed());

    // Run inference (model returns 3 outputs, we validate the primary one)
    println!("Running model inference with test input...");
    let start = Instant::now();

    let (output, _output2, _output3) = model.forward(input);

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Display output shape
    let out_shape: [usize; 3] = output.shape().dims();
    println!("\n  Model output shape: {:?}", out_shape);

    if out_shape != ref_shape {
        eprintln!(
            "FAILED: Expected output shape {:?}, got {:?}",
            ref_shape, out_shape
        );
        std::process::exit(1);
    }
    println!("  Shape matches expected: {:?}", ref_shape);

    println!("\nComparing model output with reference data...");

    let diff = output - reference_output;
    let abs_diff = diff.abs();
    let max_diff: f32 = abs_diff.clone().max().into_scalar::<f32>();
    let mean_diff: f32 = abs_diff.mean().into_scalar::<f32>();

    println!("  Maximum absolute difference: {:.6}", max_diff);
    println!("  Mean absolute difference: {:.6}", mean_diff);

    let max_diff_threshold = 1e-3;
    let mean_diff_threshold = 1e-4;
    let validation = if max_diff <= max_diff_threshold && mean_diff <= mean_diff_threshold {
        println!(
            "  Within tolerance (max<{}, mean<{})",
            max_diff_threshold, mean_diff_threshold
        );
        "Passed"
    } else {
        eprintln!(
            "  EXCEEDED tolerance (max<{}, mean<{})",
            max_diff_threshold, mean_diff_threshold
        );
        std::process::exit(1);
    };

    println!("\n========================================");
    println!("Summary:");
    println!("  - Model initialization: {:.2?}", init_time);
    println!("  - Data loading: {:.2?}", load_time);
    println!("  - Inference time: {:.2?}", inference_time);
    println!("  - Output validation: {}", validation);
    println!("========================================");
}
