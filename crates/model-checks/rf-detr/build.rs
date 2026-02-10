use burn_onnx::ModelGen;
use std::path::PathBuf;

fn artifacts_dir() -> PathBuf {
    let base = match std::env::var("BURN_CACHE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => dirs::cache_dir()
            .expect("could not determine cache directory")
            .join("burn-onnx"),
    };
    let dir = base.join("model-checks").join("rf-detr");
    println!(
        "cargo:warning=model-checks: artifacts dir = {}",
        dir.display()
    );
    dir
}

fn main() {
    let artifacts = artifacts_dir();
    let onnx_path = artifacts.join("rf_detr_small.onnx");
    let test_data_path = artifacts.join("rf_detr_small_test_data.pt");

    // Tell Cargo to only rebuild if these files change
    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed={}", test_data_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BURN_CACHE_DIR");

    // Check if the ONNX model file exists
    if !onnx_path.exists() {
        eprintln!("Error: ONNX model file not found at '{}'", onnx_path.display());
        eprintln!();
        eprintln!("Please run the following command to download and prepare the RF-DETR model:");
        eprintln!("  uv run --python 3.11 get_model.py");
        eprintln!();
        eprintln!("This will download and export the RF-DETR Small model to ONNX format.");
        std::process::exit(1);
    }

    // Generate the model code from the ONNX file
    ModelGen::new()
        .input(onnx_path.to_str().unwrap())
        .out_dir("model/")
        .run_from_script();
}
