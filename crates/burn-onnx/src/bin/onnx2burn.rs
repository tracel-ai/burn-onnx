use burn_onnx::ModelGen;

/// Takes an ONNX file and generates a model from it.
///
/// Usage: onnx2burn <input.onnx> <output_dir> [--no-simplify]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let onnx_file = args.get(1).expect("No input file provided");
    let output_dir = args.get(2).expect("No output directory provided");
    let simplify = !args.iter().any(|a| a == "--no-simplify");

    // Generate the model code from the ONNX file.
    // Weights are saved in burnpack format (.bpk file alongside the generated code)
    ModelGen::new()
        .input(onnx_file.as_str())
        .development(true)
        .simplify(simplify)
        .out_dir(output_dir.as_str())
        .run_from_cli();
}
