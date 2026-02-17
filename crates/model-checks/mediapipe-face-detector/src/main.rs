extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::PytorchStore;
use std::time::Instant;

model_checks_common::backend_type!();

// Include the generated model
include!(concat!(env!("OUT_DIR"), "/model/face_detector.rs"));

/// Test data structure matching the PyTorch saved format.
/// BlazeFace outputs two 3D tensors: regressors and classifiers.
#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input: Param<Tensor<B, 4>>,
    /// Bounding box and keypoint regressors: [1, 896, 16]
    regressors: Param<Tensor<B, 3>>,
    /// Face confidence classifiers: [1, 896, 1]
    classificators: Param<Tensor<B, 3>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        // BlazeFace Short Range: 128x128 NHWC input, 896 anchors
        Self {
            input: Initializer::Zeros.init([1, 128, 128, 3], device),
            regressors: Initializer::Zeros.init([1, 896, 16], device),
            classificators: Initializer::Zeros.init([1, 896, 1], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("MediaPipe Face Detector (BlazeFace) Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("mediapipe-face-detector");
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

    // Check if model files exist
    let model_file = artifacts_dir.join("face_detector.onnx");
    let test_data_file = artifacts_dir.join("face_detector_test_data.pt");

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
    let device = Default::default();
    let weights_path = concat!(env!("OUT_DIR"), "/model/face_detector.bpk");
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

    // Get the input tensor from test data
    let input = test_data.input.val();
    let input_shape = input.shape();
    println!("  Input shape: {:?}", input_shape.as_slice());

    // Get reference outputs
    let ref_regressors = test_data.regressors.val();
    let ref_classifiers = test_data.classificators.val();
    println!(
        "  Reference regressors shape: {:?}",
        ref_regressors.shape().as_slice()
    );
    println!(
        "  Reference classifiers shape: {:?}",
        ref_classifiers.shape().as_slice()
    );

    // Run inference
    println!("\nRunning model inference...");
    let start = Instant::now();
    let (out_regressors, out_classifiers) = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    println!("\nModel outputs:");
    println!(
        "  regressors shape: {:?}",
        out_regressors.shape().as_slice()
    );
    println!(
        "  classifiers shape: {:?}",
        out_classifiers.shape().as_slice()
    );

    // Compare outputs
    println!("\nComparing outputs with reference data...");

    let mut all_passed = true;

    for (name, output, reference) in [
        ("regressors", out_regressors, ref_regressors),
        ("classifiers", out_classifiers, ref_classifiers),
    ] {
        print!("\n  Checking {name}:");
        if output
            .clone()
            .all_close(reference.clone(), Some(1e-4), Some(1e-4))
        {
            println!(" PASS (within 1e-4)");
        } else {
            println!(" MISMATCH");
            all_passed = false;

            let diff = output.clone() - reference.clone();
            let abs_diff = diff.abs();
            let max_diff = abs_diff.clone().max().into_scalar();
            let mean_diff = abs_diff.mean().into_scalar();
            println!("    Max abs diff:  {:.6}", max_diff);
            println!("    Mean abs diff: {:.6}", mean_diff);

            println!("\n    Sample values (first 5):");
            let out_flat = output.flatten::<1>(0, 2);
            let ref_flat = reference.flatten::<1>(0, 2);
            for i in 0..5.min(out_flat.shape()[0]) {
                let m: f32 = out_flat.clone().slice(s![i..i + 1]).into_scalar();
                let r: f32 = ref_flat.clone().slice(s![i..i + 1]).into_scalar();
                println!(
                    "      [{i}] model={m:.6}, ref={r:.6}, diff={:.6}",
                    (m - r).abs()
                );
            }
        }
    }

    println!("\n========================================");
    println!("Summary:");
    println!("  Model init:  {init_time:.2?}");
    println!("  Data load:   {load_time:.2?}");
    println!("  Inference:   {inference_time:.2?}");
    if all_passed {
        println!("  Validation:  PASS");
        println!("========================================");
        println!("Model test completed successfully!");
    } else {
        println!("  Validation:  FAIL");
        println!("========================================");
        println!("Model test completed with differences.");
        std::process::exit(1);
    }
}
