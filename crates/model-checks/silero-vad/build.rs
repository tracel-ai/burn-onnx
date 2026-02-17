use burn_onnx::ModelGen;

fn main() {
    let artifacts = model_checks_common::artifacts_dir_build("silero-vad");
    let onnx_path = artifacts.join("silero_vad.onnx");

    // Tell Cargo to only rebuild if these files change
    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BURN_CACHE_DIR");

    // Check if the ONNX model file exists
    if !onnx_path.exists() {
        eprintln!(
            "Error: ONNX model file not found at '{}'",
            onnx_path.display()
        );
        eprintln!();
        eprintln!("Please run the following command to download the model:");
        eprintln!("  uv run get_model.py");
        eprintln!();
        eprintln!("This will download the Silero VAD model.");
        std::process::exit(1);
    }

    // Generate the model code from the ONNX file
    ModelGen::new()
        .input(onnx_path.to_str().unwrap())
        .out_dir("model/")
        .run_from_script();
}
