extern crate alloc;

use burn::module::{Initializer, Param};
use burn::prelude::*;

use burn_store::{ModuleSnapshot, PytorchStore};
use std::time::Instant;

model_checks_common::backend_type!();

include!(concat!(env!("OUT_DIR"), "/model_info.rs"));

use qwen_model::Model;

#[derive(Debug, Module)]
struct TestData<B: Backend> {
    input_ids: Param<Tensor<B, 2, Int>>,
    logits: Param<Tensor<B, 3>>,
}

impl<B: Backend> TestData<B> {
    fn new(device: &B::Device) -> Self {
        use burn::module::ParamId;
        Self {
            input_ids: Param::initialized(ParamId::new(), Tensor::zeros([1, SEQ_LENGTH], device)),
            logits: Initializer::Zeros.init([1, SEQ_LENGTH, VOCAB_SIZE], device),
        }
    }
}

fn get_model_display_name(model_name: &str) -> &str {
    match model_name {
        "qwen1.5-0.5b" => "Qwen1.5 0.5B",
        "qwen2.5-0.5b" => "Qwen2.5 0.5B",
        "qwen3-0.6b" => "Qwen3 0.6B",
        _ => model_name,
    }
}

fn main() {
    let model_name = MODEL_NAME;
    let display_name = get_model_display_name(model_name);

    println!("========================================");
    println!("{} Burn Model Test", display_name);
    println!("========================================\n");

    let artifacts_dir = model_checks_common::artifacts_dir("qwen");
    println!("Artifacts directory: {}", artifacts_dir.display());

    if !artifacts_dir.exists() {
        eprintln!(
            "Error: artifacts directory not found at '{}'!",
            artifacts_dir.display()
        );
        eprintln!("Please run get_model.py first to download the model and test data.");
        eprintln!("Example: uv run get_model.py --model {}", model_name);
        std::process::exit(1);
    }

    let safe_name = model_name.replace('.', "_");
    let model_file = artifacts_dir.join(format!("{}_opset16.onnx", safe_name));
    let test_data_file = artifacts_dir.join(TEST_DATA_FILE);

    if !model_file.exists() || !test_data_file.exists() {
        eprintln!("Error: Model files not found for {}!", display_name);
        eprintln!("Please run: uv run get_model.py --model {}", model_name);
        eprintln!();
        eprintln!("Available models:");
        eprintln!("  - qwen1.5-0.5b");
        eprintln!("  - qwen2.5-0.5b");
        eprintln!("  - qwen3-0.6b");
        std::process::exit(1);
    }

    println!("Initializing {} model...", display_name);
    let start = Instant::now();
    let device = Default::default();
    let model: Model<MyBackend> = Model::default();
    let init_time = start.elapsed();
    println!("  Model initialized in {:.2?}", init_time);

    let model_txt_path = artifacts_dir.join(format!("{}_model.txt", safe_name));
    println!(
        "\nSaving model structure to {}...",
        model_txt_path.display()
    );
    let model_str = format!("{}", model);
    std::fs::write(&model_txt_path, &model_str).expect("Failed to write model structure to file");
    println!("  Model structure saved");

    println!("\nLoading test data from {}...", test_data_file.display());
    let start = Instant::now();
    let mut test_data = TestData::<MyBackend>::new(&device);
    let mut store = PytorchStore::from_file(&test_data_file);
    test_data
        .load_from(&mut store)
        .expect("Failed to load test data");
    let load_time = start.elapsed();
    println!("  Data loaded in {:.2?}", load_time);

    let input_ids = test_data.input_ids.val();
    let reference_logits = test_data.logits.val();

    let input_shape: [usize; 2] = input_ids.shape().dims();
    let ref_shape: [usize; 3] = reference_logits.shape().dims();
    println!("  input_ids shape: {:?}", input_shape);
    println!("  reference logits shape: {:?}", ref_shape);

    // Warmup run (compiles GPU shaders, allocates buffers)
    println!("\nWarmup inference...");
    let start = Instant::now();
    let _ = model.forward(input_ids.clone());
    println!("  Warmup completed in {:.2?}", start.elapsed());

    println!("Running model inference...");
    let start = Instant::now();
    let output_logits = model.forward(input_ids);
    let inference_time = start.elapsed();
    println!("  Inference completed in {:.2?}", inference_time);
    let out_shape: [usize; 3] = output_logits.shape().dims();
    println!("  Output logits shape: {:?}", out_shape);

    println!("\nComparing model outputs with reference data...");

    let diff = output_logits - reference_logits;
    let abs_diff = diff.abs();
    let max_diff: f32 = abs_diff.clone().max().into_scalar();
    let mean_diff: f32 = abs_diff.mean().into_scalar();

    println!("  Maximum absolute difference: {:.6}", max_diff);
    println!("  Mean absolute difference: {:.6}", mean_diff);

    let max_diff_threshold = 1e-2;
    let mean_diff_threshold = 1e-3;
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
    println!("  - Model: {}", display_name);
    println!("  - Model initialization: {:.2?}", init_time);
    println!("  - Data loading: {:.2?}", load_time);
    println!("  - Inference time: {:.2?}", inference_time);
    println!("  - Output validation: {}", validation);
    println!("========================================");
}
