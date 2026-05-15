extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;
use burn::tensor::DType;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

// Import the generated model code as a module
pub mod sdxl_unet {
    include!(concat!(env!("OUT_DIR"), "/model/sdxl-unet.rs"));
}

/// Test data struct matching the ONNX input/output names.
///
/// All fields are stored as float tensors. The timestep (Int64 in ONNX) is
/// saved as float32 in the .pt file and cast to int before passing to forward().
#[derive(Debug, Module)]
struct TestData {
    sample: Param<Tensor<4>>,
    timestep: Param<Tensor<1>>,
    encoder_hidden_states: Param<Tensor<3>>,
    text_embeds: Param<Tensor<2>>,
    time_ids: Param<Tensor<2>>,
    out_sample: Param<Tensor<4>>,
}

impl TestData {
    fn new(device: &Device) -> Self {
        // Small latent dims for fast testing: 16x16 latent = 128x128 pixel image
        Self {
            sample: Initializer::Zeros.init([1, 4, 16, 16], device),
            timestep: Initializer::Zeros.init([1], device),
            encoder_hidden_states: Initializer::Zeros.init([1, 77, 2048], device),
            text_embeds: Initializer::Zeros.init([1, 1280], device),
            time_ids: Initializer::Zeros.init([1, 6], device),
            out_sample: Initializer::Zeros.init([1, 4, 16, 16], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("Stable Diffusion XL UNet Burn Model Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("stable-diffusion-xl");
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
    println!("Initializing SDXL UNet model...");
    let start = Instant::now();
    let device = model_checks_common::best_device!();
    let weights_path = concat!(env!("OUT_DIR"), "/model/sdxl-unet.bpk");
    let model: sdxl_unet::Model = sdxl_unet::Model::from_file(weights_path, &device);
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

    // Get input tensors
    let sample = test_data.sample.val();
    println!("  sample shape: {:?}", sample.shape().dims::<4>());

    let timestep = test_data.timestep.val();
    println!("  timestep shape: {:?}", timestep.shape().dims::<1>());

    let encoder_hidden_states = test_data.encoder_hidden_states.val();
    println!(
        "  encoder_hidden_states shape: {:?}",
        encoder_hidden_states.shape().dims::<3>()
    );

    let text_embeds = test_data.text_embeds.val();
    println!("  text_embeds shape: {:?}", text_embeds.shape().dims::<2>());

    let time_ids = test_data.time_ids.val();
    println!("  time_ids shape: {:?}", time_ids.shape().dims::<2>());

    // Get reference output
    let reference_out = test_data.out_sample.val();
    let ref_shape: [usize; 4] = reference_out.shape().dims();
    println!("  reference out_sample shape: {:?}", ref_shape);

    // Timestep is Int64 in ONNX, stored as float in test data. Cast explicitly.
    let timestep_int: Tensor<1, Int> = timestep.int().cast(DType::I64);

    // Run inference
    println!("\nRunning model inference with test input...");
    let start = Instant::now();

    let out_sample = model.forward(
        sample,
        timestep_int,
        encoder_hidden_states,
        text_embeds,
        time_ids,
    );

    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    // Verify output shape
    let out_shape: [usize; 4] = out_sample.shape().dims();
    println!("\n  Model output shape: {:?}", out_shape);

    if out_shape != ref_shape {
        eprintln!(
            "FAILED: Expected out_sample shape {:?}, got {:?}",
            ref_shape, out_shape
        );
        std::process::exit(1);
    }
    println!("  Shape matches expected: {:?}", ref_shape);

    // Compare outputs
    println!("\nComparing model outputs with reference data...");

    let diff = out_sample - reference_out;
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
