extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

// Import the generated model code as a module
pub mod depth_pro {
    include!(concat!(env!("OUT_DIR"), "/model/depth-pro.rs"));
}

#[derive(Debug, Module)]
struct TestData {
    pixel_values: Param<Tensor<4>>,
    predicted_depth: Param<Tensor<3>>,
}

impl TestData {
    fn new(device: &Device) -> Self {
        // DepthPro: input 1536x1536, output depth map 1536x1536
        Self {
            pixel_values: Initializer::Zeros.init([1, 3, 1536, 1536], device),
            predicted_depth: Initializer::Zeros.init([1, 1536, 1536], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("Apple Depth Pro Burn Model Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("depth-pro");
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
    println!("Initializing Depth Pro model...");
    let start = Instant::now();
    let device = model_checks_common::best_device!();
    let weights_path = concat!(env!("OUT_DIR"), "/model/depth-pro.bpk");
    let model: depth_pro::Model = depth_pro::Model::from_file(weights_path, &device);
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
    let pixel_values = test_data.pixel_values.val();
    let pixel_values_shape: [usize; 4] = pixel_values.shape().dims();
    println!("  Loaded pixel_values with shape: {:?}", pixel_values_shape);

    // Get the reference output
    let reference_depth = test_data.predicted_depth.val();
    let ref_depth_shape: [usize; 3] = reference_depth.shape().dims();
    println!(
        "  Loaded reference predicted_depth with shape: {:?}",
        ref_depth_shape
    );

    // Warmup run (compiles GPU shaders, allocates buffers)
    println!("\nWarmup inference...");
    let start = Instant::now();
    let _ = model.forward(pixel_values.clone());
    println!("  Warmup completed in {:.2?}", start.elapsed());

    // Run inference (model returns predicted_depth + focallength_px)
    println!("Running model inference with test input...");
    let start = Instant::now();

    let (predicted_depth_4d, _focallength_px) = model.forward(pixel_values);

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // The model returns rank 4 due to If node branch alignment; squeeze dim 0 to rank 3
    let predicted_depth: Tensor<3> = predicted_depth_4d.squeeze_dim(0);

    // Display output shape
    let depth_shape: [usize; 3] = predicted_depth.shape().dims();
    println!("\n  Model output shapes:");
    println!("    predicted_depth: {:?}", depth_shape);

    if depth_shape != ref_depth_shape {
        eprintln!(
            "FAILED: Expected predicted_depth shape {:?}, got {:?}",
            ref_depth_shape, depth_shape
        );
        std::process::exit(1);
    }
    println!("  Shape matches expected: {:?}", ref_depth_shape);

    println!("\nComparing model outputs with reference data...");

    let diff = predicted_depth - reference_depth;
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
