extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

pub mod shufflenet {
    include!(concat!(env!("OUT_DIR"), "/model/shufflenet.rs"));
}

#[derive(Debug, Module)]
struct TestData {
    input: Param<Tensor<4>>,
    output: Param<Tensor<2>>,
}

impl TestData {
    fn new(device: &Device) -> Self {
        Self {
            input: Initializer::Zeros.init([1, 3, 224, 224], device),
            output: Initializer::Zeros.init([1, 1000], device),
        }
    }
}

fn main() {
    println!("========================================");
    println!("ShuffleNet v2 x1.0 Burn Model Test");
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("shufflenet");
    println!("Artifacts directory: {}", artifacts_dir.display());

    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model and test data.");
        std::process::exit(1);
    }

    println!("Initializing ShuffleNet model...");
    let start = Instant::now();
    let device = model_checks_common::best_device!();
    let weights_path = concat!(env!("OUT_DIR"), "/model/shufflenet.bpk");
    let model: shufflenet::Model = shufflenet::Model::from_file(weights_path, &device);
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

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

    let input = test_data.input.val();
    let reference_output = test_data.output.val();
    println!("  Input shape: {:?}", input.shape().dims::<4>());
    println!(
        "  Reference output shape: {:?}",
        reference_output.shape().dims::<2>()
    );

    println!("\nWarmup inference...");
    let start = Instant::now();
    let _ = model.forward(input.clone());
    println!("  Warmup completed in {:.2?}", start.elapsed());

    println!("Running model inference...");
    let start = Instant::now();
    let output = model.forward(input);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);

    let output_shape = output.shape().dims::<2>();
    let ref_shape = reference_output.shape().dims::<2>();
    println!("\n  Output shape: {:?}", output_shape);

    if output_shape != ref_shape {
        eprintln!(
            "FAILED: Expected shape {:?}, got {:?}",
            ref_shape, output_shape
        );
        std::process::exit(1);
    }

    println!("\nComparing outputs with reference data...");
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
