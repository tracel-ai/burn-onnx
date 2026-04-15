extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::PytorchStore;
use std::time::Instant;

model_checks_common::backend_type!();

// Include the generated model
include!(concat!(env!("OUT_DIR"), "/model/arcface.rs"));

/// Test data structure matching the PyTorch saved format.
/// ArcFace takes a 112x112 face image and produces a 512-dim embedding.
#[derive(Debug, Module)]
struct TestData<B: Backend> {
    data: Param<Tensor<B, 4>>,
    fc1: Param<Tensor<B, 2>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            data: Initializer::Zeros.init([1, 3, 112, 112], device),
            fc1: Initializer::Zeros.init([1, 512], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("ArcFace (LResNet100E-IR) Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("arcface");
    println!("Artifacts directory: {}", artifacts_dir.display());

    // Check if artifacts exist
    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model.");
        eprintln!("Example: uv run get_model.py");
        std::process::exit(1);
    }

    let model_file = artifacts_dir.join("arcface.onnx");
    let test_data_file = artifacts_dir.join("test_data.pt");

    if !model_file.exists() {
        eprintln!("Error: Model file not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    if !test_data_file.exists() {
        eprintln!("Error: Test data file not found!");
        eprintln!("Please run: uv run get_model.py");
        std::process::exit(1);
    }

    // Initialize the model with weights
    println!("Initializing model...");
    let start = Instant::now();
    let device = model_checks_common::best_device!();
    let weights_path = concat!(env!("OUT_DIR"), "/model/arcface.bpk");
    let model: Model<MyBackend> = Model::from_file(weights_path, &device);
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    // Load test data from PyTorch file
    println!("\nLoading test data from {}...", test_data_file.display());
    let start = Instant::now();
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file(&test_data_file);
    test_data
        .load_from(&mut store)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    let input = test_data.data.val();
    let reference = test_data.fc1.val();
    println!("  Input shape: {:?}", input.shape().as_slice());
    println!(
        "  Reference output shape: {:?}",
        reference.shape().as_slice()
    );

    // Warmup run
    println!("\nWarmup inference...");
    let start = Instant::now();
    let _ = model.forward(input.clone());
    println!("  Warmup completed in {:.2?}", start.elapsed());

    // Run inference
    println!("Running model inference...");
    let start = Instant::now();
    let output = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    println!("\nModel output:");
    println!("  fc1 shape: {:?}", output.shape().as_slice());

    // Compare outputs
    println!("\nComparing outputs with reference data...");

    let passed = output
        .clone()
        .all_close(reference.clone(), Some(1e-4), Some(1e-4));

    println!("\n========================================");
    println!("Summary:");
    println!("  Model init:  {init_time:.2?}");
    println!("  Data load:   {load_time:.2?}");
    println!("  Inference:   {inference_time:.2?}");

    if passed {
        println!("  Validation:  PASS (within 1e-4)");
        println!("========================================");
        println!("Model test completed successfully!");
    } else {
        let diff = (output.clone() - reference.clone()).abs();
        let max_diff: f32 = diff.clone().max().into_scalar();
        let mean_diff: f32 = diff.mean().into_scalar();
        println!("  Max abs diff:  {:.6}", max_diff);
        println!("  Mean abs diff: {:.6}", mean_diff);
        println!("  Validation:  FAIL");
        println!("========================================");

        println!("\nSample values (first 5):");
        for i in 0..5 {
            let m: f32 = output.clone().slice(s![0, i..i + 1]).into_scalar();
            let r: f32 = reference.clone().slice(s![0, i..i + 1]).into_scalar();
            println!(
                "  [{i}] model={m:.6}, ref={r:.6}, diff={:.6}",
                (m - r).abs()
            );
        }

        println!("Model test completed with differences.");
        std::process::exit(1);
    }
}
