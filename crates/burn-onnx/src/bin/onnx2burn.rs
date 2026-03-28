use burn_onnx::{LoadStrategy, ModelGen};
use clap::Parser;

/// Convert an ONNX model to Burn Rust source code and `.bpk` weights.
#[derive(Parser)]
#[command(
    name = "onnx2burn",
    version,
    about = "Takes an ONNX file and generates a model from it."
)]
struct Args {
    /// Path to the input ONNX file
    input: String,
    /// Output directory for generated Rust code and weights
    output_dir: String,
    /// Disable graph simplification passes
    #[arg(long)]
    no_simplify: bool,
    /// Disable submodule partitioning for large models
    #[arg(long)]
    no_partition: bool,
    /// Disable development mode (suppresses `.onnx.txt` and `.graph.txt` debug files)
    #[arg(long)]
    no_development: bool,
    /// Embed model weights into the generated Rust code instead of a `.bpk` file
    #[arg(long)]
    embed_states: bool,
}

fn main() {
    let args = Args::parse();

    let load_strategy = if args.embed_states {
        LoadStrategy::Embedded
    } else {
        LoadStrategy::File
    };

    ModelGen::new()
        .input(&args.input)
        .out_dir(&args.output_dir)
        .development(!args.no_development)
        .simplify(!args.no_simplify)
        .partition(!args.no_partition)
        .load_strategy(load_strategy)
        .run_from_cli();
}
