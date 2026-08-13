use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    burn::custom_op::{CustomOp, HookRegistry, OpOverride},
    burn::graph::BurnGraph,
    format_tokens,
    logger::init_log,
};

use onnx_ir::{OnnxGraphBuilder, ir::OnnxGraph};

/// Controls how model weights are loaded at runtime.
///
/// This determines which constructors are generated on the `Model` struct.
/// `from_bytes()` is generated for all variants except `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadStrategy {
    /// Keep weights in a separate `.bpk` file. Generates `from_file()`, `from_bytes()`,
    /// and a `Default` impl that calls `from_file()`.
    #[default]
    File,

    /// Embed weights in the binary via `include_bytes!`. Generates `from_embedded()`,
    /// `from_bytes()`, and a `Default` impl that calls `from_embedded()`.
    Embedded,

    /// No built-in file or embedded loading. Generates only `from_bytes()`.
    /// The caller must provide weight bytes at runtime.
    Bytes,

    /// No weight-loading constructors at all.
    None,
}

/// Builder for generating Burn model code from ONNX files.
///
/// `ModelGen` converts ONNX models into Burn-compatible Rust source code and model weights.
/// It can be used from both build scripts and CLI applications.
///
/// # Conversion Process
///
/// 1. Parses ONNX model file(s)
/// 2. Converts ONNX operations to Burn nodes using the node registry
/// 3. Generates Rust source code with type-safe tensor operations
/// 4. Saves model weights in BurnPack (.bpk) format
///
/// # Examples
///
/// ## Using in a build script (`build.rs`)
///
/// ```no_run
/// use burn_onnx::ModelGen;
///
/// ModelGen::new()
///     .input("path/to/model.onnx")
///     .out_dir("model/")
///     .run_from_script();
/// ```
///
/// This generates code in `$OUT_DIR/model/model.rs` which can be included in your crate:
///
/// ```ignore
/// include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
/// ```
///
/// ## Using from CLI
///
/// ```no_run
/// use burn_onnx::ModelGen;
///
/// ModelGen::new()
///     .input("path/to/model.onnx")
///     .out_dir("src/model/")
///     .run_from_cli();
/// ```
///
/// ## Development mode for debugging
///
/// ```no_run
/// use burn_onnx::ModelGen;
///
/// ModelGen::new()
///     .input("path/to/model.onnx")
///     .out_dir("model/")
///     .development(true)  // Generates .onnx.txt and .graph.txt debug files
///     .run_from_cli();
/// ```
#[derive(Debug)]
pub struct ModelGen {
    out_dir: Option<PathBuf>,
    /// List of onnx files to generate source code from.
    inputs: Vec<PathBuf>,
    development: bool,
    load_strategy: LoadStrategy,
    /// Whether to run graph simplification passes (default: true)
    simplify: bool,
    /// Whether to partition large models into submodules (default: true)
    partition: bool,
    /// User hooks for custom (non-built-in) ops. Shared with the onnx-ir
    /// parse pipeline (type inference) and the codegen dispatch.
    hooks: Arc<HookRegistry>,
}

impl Default for ModelGen {
    fn default() -> Self {
        Self {
            out_dir: None,
            inputs: Vec::new(),
            development: false,
            load_strategy: LoadStrategy::default(),
            simplify: true,
            partition: true,
            hooks: Arc::new(HookRegistry::default()),
        }
    }
}

impl ModelGen {
    /// Creates a new `ModelGen` builder with default settings.
    ///
    /// Default configuration:
    /// - Development mode: off
    /// - Load strategy: [`LoadStrategy::File`]
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .input("model.onnx")
    ///     .out_dir("./out")
    ///     .run_from_cli();
    /// ```
    pub fn new() -> Self {
        init_log().ok(); // Error when init multiple times are ignored.
        Self::default()
    }

    /// Sets the output directory for generated files.
    ///
    /// When used with [`run_from_script`](Self::run_from_script), this path is appended to
    /// `$OUT_DIR`. When used with [`run_from_cli`](Self::run_from_cli), this is the absolute
    /// or relative path where files will be written.
    ///
    /// # Arguments
    ///
    /// * `out_dir` - Directory path where generated `.rs` and record files will be saved
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .out_dir("model/")  // In build.rs: $OUT_DIR/model/
    ///     .input("model.onnx")
    ///     .run_from_script();
    /// ```
    pub fn out_dir(&mut self, out_dir: &str) -> &mut Self {
        self.out_dir = Some(Path::new(out_dir).into());
        self
    }

    /// Adds an ONNX model file to convert.
    ///
    /// Multiple input files can be added by calling this method multiple times.
    /// Each input file will generate a separate `.rs` file with the same base name.
    ///
    /// # Arguments
    ///
    /// * `input` - Path to the ONNX model file (`.onnx`)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .input("encoder.onnx")
    ///     .input("decoder.onnx")  // Generate multiple models
    ///     .out_dir("models/")
    ///     .run_from_cli();
    /// ```
    pub fn input(&mut self, input: &str) -> &mut Self {
        self.inputs.push(input.into());
        self
    }

    /// Enables development mode for debugging.
    ///
    /// When enabled, generates additional debug files alongside the Rust source:
    /// - `<model>.onnx.txt` - Debug representation of the parsed ONNX graph
    /// - `<model>.graph.txt` - Debug representation of the converted Burn graph
    ///
    /// # Arguments
    ///
    /// * `development` - If `true`, generate debug files
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .input("model.onnx")
    ///     .out_dir("debug/")
    ///     .development(true)  // Generates model.onnx.txt and model.graph.txt
    ///     .run_from_cli();
    /// ```
    pub fn development(&mut self, development: bool) -> &mut Self {
        self.development = development;
        self
    }

    /// Sets the weight loading strategy for the generated model.
    ///
    /// See [`LoadStrategy`] for available options.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::{ModelGen, LoadStrategy};
    ///
    /// // WASM or embedded: load weights from bytes at runtime
    /// ModelGen::new()
    ///     .input("model.onnx")
    ///     .out_dir("model/")
    ///     .load_strategy(LoadStrategy::Bytes)
    ///     .run_from_script();
    /// ```
    pub fn load_strategy(&mut self, strategy: LoadStrategy) -> &mut Self {
        self.load_strategy = strategy;
        self
    }

    /// Enable or disable graph simplification passes (default: true).
    ///
    /// When enabled, optimization passes like dead node elimination, common
    /// subexpression elimination (CSE), and pattern-based simplifications
    /// are applied to the ONNX IR before code generation.
    pub fn simplify(&mut self, simplify: bool) -> &mut Self {
        self.simplify = simplify;
        self
    }

    /// Enable or disable submodule partitioning for large models (default: true).
    ///
    /// When enabled, models with more than 200 nodes are automatically split into
    /// smaller submodule structs to keep generated code compilable. Each submodule
    /// gets its own `forward()` method, and the top-level `Model` delegates to them.
    pub fn partition(&mut self, partition: bool) -> &mut Self {
        self.partition = partition;
        self
    }

    /// Register a codegen hook for a custom (non-built-in) ONNX operator.
    ///
    /// The hook is matched by ONNX operator identity `(op_type, domain)` and
    /// supplies both type inference (during parsing) and code generation for
    /// matching nodes. See [`crate::ext::CustomOp`].
    ///
    /// # Panics
    ///
    /// Panics if a hook for the same `(op_type, domain)` is already registered.
    pub fn register_custom_op(&mut self, op: impl CustomOp) -> &mut Self {
        Arc::get_mut(&mut self.hooks)
            .expect("register_custom_op must be called before running the generation")
            .add_custom_op(Box::new(op));
        self
    }

    /// Register a codegen override for a built-in ONNX operator.
    ///
    /// The built-in processor still performs type inference; the override
    /// replaces only the generated code for nodes of the target type. See
    /// [`crate::ext::OpOverride`].
    ///
    /// # Panics
    ///
    /// Panics if an override for the same target is already registered, or if
    /// the target is `NodeType::Custom`.
    pub fn register_op_override(&mut self, over: impl OpOverride) -> &mut Self {
        Arc::get_mut(&mut self.hooks)
            .expect("register_op_override must be called before running the generation")
            .add_override(Box::new(over));
        self
    }

    /// Runs code generation from a build script context.
    ///
    /// Use this method when calling from `build.rs`. The output directory will be
    /// `$OUT_DIR/<out_dir>`, allowing the generated code to be included with:
    ///
    /// ```ignore
    /// include!(concat!(env!("OUT_DIR"), "/<out_dir>/<model>.rs"));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `OUT_DIR` environment variable is not set (should be set by Cargo).
    ///
    /// # Examples
    ///
    /// In `build.rs`:
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .input("path/to/model.onnx")
    ///     .out_dir("model/")
    ///     .run_from_script();
    /// ```
    pub fn run_from_script(&self) {
        self.run(true);
    }

    /// Runs code generation from a CLI or application context.
    ///
    /// Use this method when calling from a CLI tool or regular application.
    /// The output directory is used as-is (relative or absolute path).
    ///
    /// # Panics
    ///
    /// Panics if `out_dir` was not set via [`out_dir`](Self::out_dir).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use burn_onnx::ModelGen;
    ///
    /// ModelGen::new()
    ///     .input("model.onnx")
    ///     .out_dir("./generated/")
    ///     .run_from_cli();
    /// ```
    pub fn run_from_cli(&self) {
        self.run(false);
    }

    /// Run code generation.
    fn run(&self, is_build_script: bool) {
        log::info!("Starting to convert ONNX to Burn");

        let out_dir = self.get_output_directory(is_build_script);
        log::debug!("Output directory: {out_dir:?}");

        create_dir_all(&out_dir).unwrap();

        for input in self.inputs.iter() {
            let file_name = input.file_name().unwrap();
            let out_file: PathBuf = out_dir.join(file_name);

            log::info!("Converting {input:?}");
            log::debug!("Input file name: {file_name:?}");
            log::debug!("Output file: {out_file:?}");

            self.generate_model(input, out_file);
        }

        log::info!("Finished converting ONNX to Burn");
    }

    /// Get the output directory path based on whether this is a build script or CLI invocation.
    fn get_output_directory(&self, is_build_script: bool) -> PathBuf {
        if is_build_script {
            let cargo_out_dir = env::var("OUT_DIR").expect("OUT_DIR env is not set");
            let mut path = PathBuf::from(cargo_out_dir);
            // Append the out_dir to the cargo_out_dir
            path.push(self.out_dir.as_ref().unwrap());
            path
        } else {
            self.out_dir.as_ref().expect("out_dir is not set").clone()
        }
    }

    /// Generate model source code and model state.
    fn generate_model(&self, input: &PathBuf, out_file: PathBuf) {
        log::info!("Generating model from {input:?}");
        log::debug!("Development mode: {:?}", self.development);
        log::debug!("Output file: {out_file:?}");

        // The registry is passed even when empty: with hooks present (any
        // registry at all), the pipeline's coverage pre-pass reports ALL
        // uncovered custom ops in one friendly summary instead of failing
        // deep inside type inference.
        let graph = OnnxGraphBuilder::new()
            .simplify(self.simplify)
            .with_custom_op_inference(self.hooks.clone())
            .parse_file(input)
            .unwrap_or_else(|e| panic!("{}", parse_error_message(input, &e)));

        if self.development {
            self.write_debug_file(&out_file, "onnx.txt", &graph);
        }

        let graph = ParsedOnnxGraph(graph);

        let top_comment = Some(format!("Generated from ONNX {input:?} by burn-onnx"));

        let code = self.generate_burn_graph(graph, &out_file, top_comment);

        let code_str = format_tokens(code);
        let source_code_file = out_file.with_extension("rs");
        log::info!("Writing source code to {}", source_code_file.display());
        fs::write(source_code_file, code_str).unwrap();

        log::info!("Model generated");
    }

    /// Write debug file in development mode.
    fn write_debug_file<T: std::fmt::Debug>(&self, out_file: &Path, extension: &str, content: &T) {
        let debug_content = format!("{content:#?}");
        let debug_file = out_file.with_extension(extension);
        log::debug!("Writing debug file: {debug_file:?}");
        fs::write(debug_file, debug_content).unwrap();
    }

    /// Generate BurnGraph and codegen.
    fn generate_burn_graph(
        &self,
        graph: ParsedOnnxGraph,
        out_file: &Path,
        top_comment: Option<String>,
    ) -> proc_macro2::TokenStream {
        let bpk_file = out_file.with_extension("bpk");
        graph
            .into_burn()
            .with_hooks(self.hooks.clone())
            .with_burnpack(bpk_file, self.load_strategy)
            .with_blank_space(true)
            .with_top_comment(top_comment)
            .with_partition(self.partition)
            .codegen()
    }
}

/// User-facing message for a parse failure, with hook-registration guidance
/// appended for the missing-hook case (the onnx-ir error text itself does not
/// name ModelGen).
fn parse_error_message(input: &Path, error: &onnx_ir::Error) -> String {
    let base = format!("Failed to parse ONNX file '{}': {}", input.display(), error);
    match error {
        onnx_ir::Error::MissingCustomOpHooks(_) => {
            format!("{base}\nRegister hooks via ModelGen::register_custom_op.")
        }
        _ => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::{MissingHook, MissingReason};

    #[test]
    fn missing_hook_parse_error_appends_registration_hint() {
        let error = onnx_ir::Error::MissingCustomOpHooks(vec![MissingHook::new(
            "SelectiveScan",
            "mamba",
            12,
            3,
            MissingReason::NoHook,
        )]);
        let msg = parse_error_message(Path::new("mamba.onnx"), &error);
        assert!(msg.contains("mamba::SelectiveScan"), "got: {msg}");
        assert!(msg.contains("used by 12 node(s)"), "got: {msg}");
        assert!(msg.contains("ModelGen::register_custom_op"), "got: {msg}");
    }

    #[test]
    fn other_parse_errors_have_no_hook_hint() {
        let error = onnx_ir::Error::MissingOpsetVersion;
        let msg = parse_error_message(Path::new("m.onnx"), &error);
        assert!(!msg.contains("register_custom_op"), "got: {msg}");
    }
}

#[derive(Debug)]
struct ParsedOnnxGraph(OnnxGraph);

impl ParsedOnnxGraph {
    /// Converts ONNX graph to Burn graph.
    pub fn into_burn(self) -> BurnGraph {
        let mut graph = BurnGraph::default();

        for node in self.0.nodes {
            // Register node directly (control flow nodes will fail at codegen time)
            graph.register(node);
        }

        // Extract input and output names
        let input_names: Vec<_> = self
            .0
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();
        let output_names: Vec<_> = self
            .0
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        // Register inputs and outputs with the graph (pass Arguments directly)
        graph.register_input_output(input_names, output_names, &self.0.inputs, &self.0.outputs);

        graph
    }
}
