use burn_onnx::ModelGen;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let supported_models = vec!["qwen1.5-0.5b", "qwen2.5-0.5b", "qwen3-0.6b"];

    let model_name = env::var("QWEN_MODEL").unwrap_or_else(|_| {
        eprintln!("Error: QWEN_MODEL environment variable is not set.");
        eprintln!();
        eprintln!("Please specify which Qwen model to build:");
        eprintln!("  QWEN_MODEL=qwen2.5-0.5b cargo build");
        eprintln!();
        eprintln!("Available models: {}", supported_models.join(", "));
        std::process::exit(1);
    });

    if !supported_models.contains(&model_name.as_str()) {
        eprintln!(
            "Error: Unsupported model '{}'. Supported models: {:?}",
            model_name, supported_models
        );
        std::process::exit(1);
    }

    // Dots in model names break Rust's Path::with_extension (e.g. "qwen2.5-0.5b"
    // gets truncated). Use underscores in filenames.
    let safe_name = model_name.replace('.', "_");

    let artifacts = model_checks_common::artifacts_dir_build("qwen");
    let onnx_path = artifacts.join(format!("{}_opset16.onnx", safe_name));
    let test_data_path = artifacts.join(format!("{}_test_data.pt", safe_name));

    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed={}", test_data_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=QWEN_MODEL");
    println!("cargo:rerun-if-env-changed=BURN_CACHE_DIR");

    if !onnx_path.exists() {
        eprintln!(
            "Error: ONNX model file not found at '{}'",
            onnx_path.display()
        );
        eprintln!();
        eprintln!(
            "Please run the following command to download and prepare the {} model:",
            model_name
        );
        eprintln!("  uv run get_model.py --model {}", model_name);
        eprintln!();
        eprintln!("Available models: {}", supported_models.join(", "));
        std::process::exit(1);
    }

    ModelGen::new()
        .input(onnx_path.to_str().unwrap())
        .out_dir("model/")
        .run_from_script();

    // Model-specific configuration
    let (seq_length, vocab_size) = match model_name.as_str() {
        "qwen1.5-0.5b" => (32usize, 151936usize),
        "qwen2.5-0.5b" => (32, 151936),
        "qwen3-0.6b" => (32, 151936),
        _ => unreachable!(),
    };

    let out_dir = env::var("OUT_DIR").unwrap();
    let model_info_path = Path::new(&out_dir).join("model_info.rs");

    let model_include = format!(
        "include!(concat!(env!(\"OUT_DIR\"), \"/model/{}_opset16.rs\"));",
        safe_name
    );

    fs::write(
        model_info_path,
        format!(
            r#"pub const MODEL_NAME: &str = "{}";
pub const TEST_DATA_FILE: &str = "{}_test_data.pt";
pub const SEQ_LENGTH: usize = {};
pub const VOCAB_SIZE: usize = {};

pub mod qwen_model {{
    {}
}}"#,
            model_name, safe_name, seq_length, vocab_size, model_include
        ),
    )
    .expect("Failed to write model info");
}
