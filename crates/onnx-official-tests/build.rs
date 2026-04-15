//! Build script: drives codegen and harness generation for every
//! upstream ONNX backend node test vendored under `vendor/node/`.
//!
//! # Data flow
//!
//! ```text
//!   vendor/node/test_<name>/model.onnx   vendor/node/test_<name>/test_data_set_0/*.pb
//!              │                                          │
//!              │ (pass-list filter from expectations.toml) │
//!              ▼                                          │
//!   $OUT_DIR/staged/<name>.onnx                           │
//!              │                                          │
//!              │ ModelGen                                 │
//!              ▼                                          │
//!   $OUT_DIR/model/<name>.rs + <name>.bpk                 │
//!              │                                          │
//!              │ model.onnx introspection (rank + dtype)  │
//!              ▼                                          │
//!   $OUT_DIR/models.rs      ─── tests/test_mod.rs         │
//!   $OUT_DIR/harness.rs     ─── tests/test_mod.rs  ◀──────┘
//!   $OUT_DIR/manifest.rs    ─── tests/test_mod.rs
//! ```
//!
//! Staging exists because `burn_onnx::ModelGen` derives output filenames
//! from the input file stem, so feeding it 1500 files all named
//! `model.onnx` would collide in a single `model.rs`. Copying each to
//! `<name>.onnx` first gives every generated module a unique name that
//! matches its upstream test. (This was inherited from M1 and is
//! still the simplest workaround.)
//!
//! # Pass-list discipline
//!
//! Only tests with `status = "pass"` in `expectations.toml` are fed to
//! `ModelGen`. Any `skip-codegen` / `fail-compare` entry is read purely
//! as documentation — the build script never attempts to codegen it and
//! never emits a `#[test]` function for it. This is the "option (c)"
//! trade-off from the M2 design discussion: a mislabeled pass entry
//! that actually panics in codegen will fail the build with a path-
//! qualified message pointing at the entry, and the contributor fixes
//! the expectations row before re-running. The alternative (wrapping
//! `ModelGen` in `catch_unwind`) adds noisy warnings that hide real
//! regressions.
//!
//! # Introspection
//!
//! For each pass entry, the script reads `model.onnx` as a raw
//! `ModelProto` (re-exported from `onnx_ir`) and walks
//! `graph.input` / `graph.output` to extract `(name, rank, dtype)`
//! tuples. Initializer names are filtered out of `graph.input` because
//! `burn-onnx` turns them into module fields, not `forward` parameters.
//!
//! Any test whose introspection hits an edge case the generated harness
//! can't handle — dynamic/symbolic shape, scalar (rank-0) I/O, an
//! unsupported dtype, or a non-tensor type (sequence/map/sparse) — gets
//! a `cargo:warning=` diagnostic and is skipped from the harness. Its
//! `Model` struct is still generated and compiled; it just has no
//! `#[test]` driver pointing at it. That keeps the build green while
//! surfacing which tests need additional harness work.

use burn_onnx::ModelGen;
use onnx_ir::ModelProto;
use protobuf::Message;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

// --- Expectations parsing -----------------------------------------------
//
// build.rs only cares about the `status` field per entry, so this is a
// deliberately narrow sub-schema of the one in `src/expectations.rs`.
// Duplicating the parse avoids pulling the crate's lib into the build
// graph (which would be a circular `[dependencies]` + `[build-dependencies]`
// edge).

#[derive(Debug, Deserialize)]
struct ExpectationRow {
    status: String,
}

/// Load `expectations.toml` and return a name -> status map. Only the
/// `status` field is read; other fields (reason / tracking / wontfix)
/// are documentation for humans and are ignored here.
fn load_expectations(path: &Path) -> BTreeMap<String, String> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let raw: BTreeMap<String, ExpectationRow> =
        toml::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    raw.into_iter().map(|(k, v)| (k, v.status)).collect()
}

// --- Dtype mapping ------------------------------------------------------
//
// Mirror of `src/pb_loader.rs::TensorValues` restricted to the variants
// the harness knows how to feed into a burn-onnx-generated `forward`.
// Everything else surfaces as `introspect` returning `None`.

#[derive(Clone, Copy, Debug)]
enum Dtype {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl Dtype {
    /// Map an `onnx.TensorProto.DataType` i32 to our narrowed enum.
    /// Returns `None` for dtypes the harness can't load (FLOAT16,
    /// BFLOAT16, FLOAT8*, INT4, UINT4, FLOAT4E2M1, STRING, COMPLEX*).
    fn from_elem_type(t: i32) -> Option<Self> {
        match t {
            1 => Some(Self::F32),  // FLOAT
            2 => Some(Self::U8),   // UINT8
            3 => Some(Self::I8),   // INT8
            4 => Some(Self::U16),  // UINT16
            5 => Some(Self::I16),  // INT16
            6 => Some(Self::I32),  // INT32
            7 => Some(Self::I64),  // INT64
            9 => Some(Self::Bool), // BOOL
            11 => Some(Self::F64), // DOUBLE
            12 => Some(Self::U32), // UINT32
            13 => Some(Self::U64), // UINT64
            _ => None,
        }
    }

    /// The `TensorValues::<variant>` name used to destructure a
    /// loaded `ReferenceTensor` in the harness.
    fn values_variant(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::Bool => "Bool",
        }
    }

    /// ONNX dtype display name for diagnostic messages.
    fn onnx_name(self) -> &'static str {
        match self {
            Self::F32 => "FLOAT",
            Self::F64 => "DOUBLE",
            Self::I8 => "INT8",
            Self::I16 => "INT16",
            Self::I32 => "INT32",
            Self::I64 => "INT64",
            Self::U8 => "UINT8",
            Self::U16 => "UINT16",
            Self::U32 => "UINT32",
            Self::U64 => "UINT64",
            Self::Bool => "BOOL",
        }
    }

    /// Suffix to emit inside `Tensor<B, D, ???>` — empty for the
    /// default `Float` kind, explicit for `Int` / `Bool`. Burn groups
    /// every integer dtype under the `Int` tensor kind and uses a
    /// runtime `.cast(DType::…)` to preserve the ONNX bit width.
    fn tensor_kind_suffix(self) -> &'static str {
        match self {
            Self::F32 | Self::F64 => "",
            Self::Bool => ", burn::tensor::Bool",
            _ => ", burn::tensor::Int",
        }
    }

    /// The concrete `burn::tensor::DType` literal for this ONNX dtype.
    ///
    /// Used when emitting `Tensor::from_data(data, (device, DType::…))` so
    /// the harness pins the tensor's runtime dtype to what the `.pb` source
    /// carries, instead of letting `from_data(data, &device)` silently
    /// convert to the backend's default IntElem / FloatElem (which varies
    /// across backends). A bare `from_data` would otherwise make an I64
    /// source round-trip through a backend-default int width and then fail
    /// the later `assert_eq` against the I64 expected-value TensorData.
    fn burn_dtype_tokens(self) -> &'static str {
        match self {
            Self::F32 => "burn::tensor::DType::F32",
            Self::F64 => "burn::tensor::DType::F64",
            Self::I8 => "burn::tensor::DType::I8",
            Self::I16 => "burn::tensor::DType::I16",
            Self::I32 => "burn::tensor::DType::I32",
            Self::I64 => "burn::tensor::DType::I64",
            Self::U8 => "burn::tensor::DType::U8",
            Self::U16 => "burn::tensor::DType::U16",
            Self::U32 => "burn::tensor::DType::U32",
            Self::U64 => "burn::tensor::DType::U64",
            Self::Bool => "burn::tensor::DType::Bool(burn::tensor::BoolStore::Native)",
        }
    }

    /// The concrete Rust element type to use as the type parameter for
    /// `TensorData::assert_approx_eq::<T>`. Using `f32` for FLOAT and
    /// `f64` for DOUBLE avoids two problems: (a) a backend default
    /// `FloatElem<TestBackend>` that differs from the TensorData dtype
    /// would cause a type mismatch, and (b) using a narrower type than
    /// the decoded values would compare at reduced precision.
    fn assert_elem_type(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            _ => "f32", // unreachable for non-float; caller guards with is_float()
        }
    }

    /// Whether output comparison should use an approximate float
    /// tolerance (`assert_approx_eq`) or exact equality (`assert_eq`).
    fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

// --- Introspection result ----------------------------------------------

#[derive(Debug)]
struct IoDesc {
    rank: usize,
    dtype: Dtype,
}

#[derive(Debug)]
struct TestMeta {
    inputs: Vec<IoDesc>,
    outputs: Vec<IoDesc>,
}

/// Why a pass-listed test was skipped from harness generation. Logged
/// as a `cargo:warning=` so the contributor sees exactly which entries
/// need follow-up work.
enum SkipReason {
    Io(String),
    NotATensor,
    DynamicShape,
    UnsupportedDtype(i32),
    RankZero,
    NoInputs,
    NoOutputs,
    ParseFailure,
    MissingGraph,
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error reading model.onnx: {e}"),
            Self::NotATensor => write!(f, "input/output is not a tensor (sequence/map/sparse)"),
            Self::DynamicShape => write!(f, "dynamic/symbolic dimension in input/output shape"),
            Self::UnsupportedDtype(t) => {
                write!(
                    f,
                    "unsupported element dtype {t} (not loadable by pb_loader)"
                )
            }
            Self::RankZero => write!(f, "rank-0 (scalar) input/output not yet handled by harness"),
            Self::NoInputs => write!(f, "model has zero runtime inputs"),
            Self::NoOutputs => write!(f, "model has zero outputs"),
            Self::ParseFailure => write!(f, "protobuf decode of model.onnx failed"),
            Self::MissingGraph => write!(f, "ModelProto has no graph"),
        }
    }
}

/// Parse `model.onnx` into a `TestMeta` with per-input and per-output
/// rank+dtype info, or return the reason this test can't be wired into
/// the data-driven harness.
fn introspect(model_path: &Path) -> Result<TestMeta, SkipReason> {
    let bytes = fs::read(model_path).map_err(|e| SkipReason::Io(e.to_string()))?;
    let proto = ModelProto::parse_from_bytes(&bytes).map_err(|_| SkipReason::ParseFailure)?;

    // ModelProto.graph is a MessageField<GraphProto>; `.as_ref()` gives
    // Option<&GraphProto> and is None iff the field is absent.
    let graph = proto.graph.as_ref().ok_or(SkipReason::MissingGraph)?;

    // burn-onnx rolls initializers into module fields rather than
    // forward parameters, so we drop them from the runtime-inputs list.
    // Older ONNX files duplicate initializer names in graph.input; new
    // ones usually don't. Filtering unconditionally is safe.
    let initializer_names: BTreeSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();

    let mut inputs = Vec::new();
    for info in &graph.input {
        if initializer_names.contains(info.name.as_str()) {
            continue;
        }
        inputs.push(io_desc_from_value_info(info)?);
    }

    let mut outputs = Vec::new();
    for info in &graph.output {
        outputs.push(io_desc_from_value_info(info)?);
    }

    if inputs.is_empty() {
        return Err(SkipReason::NoInputs);
    }
    if outputs.is_empty() {
        return Err(SkipReason::NoOutputs);
    }

    // Rank-0 I/O collides with burn's scalar-vs-tensor distinction;
    // burn-onnx may emit `ScalarNative` or `ScalarTensor` instead of
    // `Tensor<B, 0>` depending on the op, and the harness doesn't yet
    // cover that branch. Skip these loudly so we know which tests to
    // come back to.
    if inputs.iter().any(|i| i.rank == 0) || outputs.iter().any(|o| o.rank == 0) {
        return Err(SkipReason::RankZero);
    }

    Ok(TestMeta { inputs, outputs })
}

/// Extract `(rank, dtype)` from a single `ValueInfoProto`.
///
/// Walks the chain `ValueInfoProto -> TypeProto -> type_proto::Tensor
/// -> TensorShapeProto -> Dimension`. The inner two types are private
/// to the re-exported `onnx_ir::protos` module, but we only ever access
/// their public fields via method calls on references we already hold,
/// which Rust allows without requiring the type names to be in scope.
///
/// Returns `Err(SkipReason::*)` for any edge case the harness can't
/// emit glue for: sparse / sequence / map types, dynamic or negative
/// dimensions, and dtypes outside the `pb_loader`-supported set.
fn io_desc_from_value_info(info: &onnx_ir::ValueInfoProto) -> Result<IoDesc, SkipReason> {
    // ValueInfoProto.type_ is a `MessageField<TypeProto>`; `as_ref`
    // flattens it into `Option<&TypeProto>`. Missing means the file is
    // malformed, which is rare enough that we treat it as a harness-
    // level skip rather than a build-wide panic.
    let type_ = info.type_.as_ref().ok_or(SkipReason::NotATensor)?;

    // TypeProto has several oneof variants (tensor / sequence / map /
    // sparse / optional). Only the tensor variant is harness-
    // expressible. `has_tensor_type` returns false for every other
    // variant and for a default-constructed TypeProto.
    if !type_.has_tensor_type() {
        return Err(SkipReason::NotATensor);
    }
    let tt = type_.tensor_type();

    let dtype =
        Dtype::from_elem_type(tt.elem_type).ok_or(SkipReason::UnsupportedDtype(tt.elem_type))?;

    // type_proto::Tensor.shape is another MessageField; a missing
    // shape means the producer declined to specify one (legitimate in
    // ONNX but useless for the harness since we need a concrete rank
    // at emit time).
    let shape = tt.shape.as_ref().ok_or(SkipReason::DynamicShape)?;

    let mut rank = 0usize;
    for dim in &shape.dim {
        // Dimension is a oneof of `DimValue(i64)` or `DimParam(String)`.
        // Anything that isn't a concrete non-negative DimValue is
        // symbolic and can't be baked into a `Tensor<B, D>` literal.
        if !dim.has_dim_value() {
            return Err(SkipReason::DynamicShape);
        }
        let v = dim.dim_value();
        if v < 0 {
            return Err(SkipReason::DynamicShape);
        }
        rank += 1;
    }

    Ok(IoDesc { rank, dtype })
}

fn main() {
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=expectations.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest_dir.join("vendor/node");
    let expectations_path = manifest_dir.join("expectations.toml");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let staged = out_dir.join("staged");
    fs::create_dir_all(&staged).unwrap();

    let expectations = load_expectations(&expectations_path);

    // Split the expectations into passes (codegen + harness) and
    // non-passes (documentation only). Drift between vendor/ and
    // expectations.toml is caught by the runtime test, not the build.
    let mut pass_names: Vec<String> = expectations
        .iter()
        .filter(|(_, s)| s.as_str() == "pass")
        .map(|(k, _)| k.clone())
        .collect();
    pass_names.sort();

    // For each pass entry: stage, introspect, and collect metadata for
    // harness emission. Tests that introspect cleanly go into the
    // harness; everything else gets a cargo warning and is dropped from
    // the harness (but still codegened so regressions surface).
    let mut model_gen = ModelGen::new();
    let mut harnessable: BTreeMap<String, TestMeta> = BTreeMap::new();
    let mut codegen_only: Vec<String> = Vec::new();

    for name in &pass_names {
        let src = vendor.join(name).join("model.onnx");
        if !src.exists() {
            println!(
                "cargo:warning=expectations.toml lists `{name}` as pass but \
                 {} is missing; skipping from codegen",
                src.display()
            );
            continue;
        }

        let dst = staged.join(format!("{name}.onnx"));
        if let Err(e) = fs::copy(&src, &dst) {
            panic!("copy {} -> {}: {e}", src.display(), dst.display());
        }
        model_gen.input(dst.to_str().expect("non-utf8 staged path"));

        match introspect(&src) {
            Ok(meta) => {
                harnessable.insert(name.clone(), meta);
            }
            Err(reason) => {
                println!("cargo:warning=skipping harness emission for {name}: {reason}");
                codegen_only.push(name.clone());
            }
        }
    }

    // Run ModelGen over every staged file. A panic here means a
    // pass-listed test actually fails codegen — the fix is to update
    // its entry in expectations.toml to `skip-codegen`, not to swallow
    // the error. (See the `build.rs` module-level docs for the
    // rationale behind this strict posture.)
    model_gen.out_dir("model/").run_from_script();

    // Post-ModelGen harness filter.
    //
    // burn-onnx's generated `forward` doesn't always return `Tensor<B,
    // D, K>`. Specific ops (ONNX `Shape`, `Size`, etc.) lower to
    // compile-time native Rust types like `[i64; N]` because burn-onnx
    // has the shape information statically. The harness only knows how
    // to call `.to_data()` on burn `Tensor`s, so any test whose
    // generated forward returns a non-Tensor type has to be skipped
    // from `#[test]` emission.
    //
    // We detect this by reading the generated `<name>.rs` and looking
    // for telltale non-Tensor patterns in the forward return type. A
    // structured parse of the Rust source would be more principled but
    // also dramatically more work; the heuristic here is liberal (err
    // on the side of skipping) and the set of affected ops is small.
    let model_out_dir = out_dir.join("model");
    let non_tensor_forwards: Vec<String> = harnessable
        .keys()
        .filter(|name| !forward_signature_is_harnessable(&model_out_dir.join(format!("{name}.rs"))))
        .cloned()
        .collect();
    for name in non_tensor_forwards {
        println!(
            "cargo:warning=skipping harness emission for {name}: forward returns a non-tensor type (e.g. [i64; N] from ONNX Shape/Size)"
        );
        harnessable.remove(&name);
        codegen_only.push(name);
    }

    // Emit the generated files that tests/test_mod.rs will pick up.
    emit_models_rs(&out_dir, &harnessable, &codegen_only);
    emit_harness_rs(&out_dir, &harnessable);
    emit_manifest_rs(&out_dir, &harnessable, &codegen_only);
}

/// Return `true` if the generated `forward` signature returns a shape
/// the data-driven harness can compare via `.to_data()`.
///
/// This is a deliberately cheap textual check: it finds the first
/// `pub fn forward` in the file, extracts the substring between `->`
/// and ` {`, and rejects obvious non-tensor patterns (compile-time
/// arrays like `[i64; N]`, bare scalar types, etc.). If the file is
/// missing or unreadable we return `true` and let the downstream
/// compile catch the problem so the build doesn't silently drop tests.
fn forward_signature_is_harnessable(generated_rs: &Path) -> bool {
    let Ok(text) = fs::read_to_string(generated_rs) else {
        return true;
    };
    let Some(idx) = text.find("pub fn forward") else {
        return true;
    };
    let slice = &text[idx..];
    let Some(arrow) = slice.find("->") else {
        return true;
    };
    // Find the opening brace of the function body. We look for `{`
    // after the arrow, skipping any amount of whitespace (or a `where`
    // clause). This is more robust than requiring the exact substring
    // `" {"` which would break if the code formatter or a future
    // burn-onnx version reformats the signature across multiple lines.
    let after_arrow = &slice[arrow + 2..];
    let brace_pos = after_arrow.find('{').unwrap_or(after_arrow.len());
    // Strip a trailing `where` clause if present — the return type
    // sits between `->` and either `where` or `{`, whichever comes
    // first.
    let mut return_type = &after_arrow[..brace_pos];
    if let Some(where_pos) = return_type.find("where") {
        return_type = &return_type[..where_pos];
    }
    let return_type = return_type.trim();

    // Patterns we know break the harness' `.to_data()` call:
    //   - `[i64; N]`, `[i32; N]`, `[f32; N]`, `[f64; N]`, etc.
    //     (burn-onnx's ONNX-`Shape`/`Size` lowering to a native array)
    //   - Tuples containing any of the above.
    //   - Bare scalar return types (`i64`, `f32`, etc.) from scalar-
    //     output ops; rare but possible.
    //
    // The check is an `||` fan-out because it's easier to extend than
    // a single regex, and a handful of passes over a ~4 KB string is
    // negligible compared to the ModelGen call that produced it.
    if return_type.contains("[i64;")
        || return_type.contains("[i32;")
        || return_type.contains("[i8;")
        || return_type.contains("[u64;")
        || return_type.contains("[u32;")
        || return_type.contains("[u8;")
        || return_type.contains("[f32;")
        || return_type.contains("[f64;")
        || return_type.contains("[bool;")
    {
        return false;
    }

    // Bare scalar returns (not wrapped in Tensor<>) are rare but
    // would also trip the harness. We require at least one `Tensor<`
    // occurrence in the return type to consider it harnessable.
    if !return_type.contains("Tensor<") {
        return false;
    }

    true
}

/// Emit `$OUT_DIR/models.rs`: one `pub mod <name>` per generated model.
/// This is include!'d as `pub mod generated` by tests/test_mod.rs so
/// the test functions can refer to `generated::test_add::Model`.
fn emit_models_rs(
    out_dir: &Path,
    harnessable: &BTreeMap<String, TestMeta>,
    codegen_only: &[String],
) {
    let mut buf = String::new();
    buf.push_str(
        "// GENERATED by build.rs — do not edit. One module per test that\n\
         // successfully went through ModelGen.\n\n",
    );
    // Both harnessable and codegen-only tests need their generated
    // module declared so `rustc` compiles the `Model` struct. The
    // difference is that codegen-only tests don't appear in harness.rs.
    let mut all_mods: Vec<&str> = harnessable
        .keys()
        .map(String::as_str)
        .chain(codegen_only.iter().map(String::as_str))
        .collect();
    all_mods.sort();
    for name in all_mods {
        writeln!(
            buf,
            "#[allow(clippy::type_complexity, non_snake_case, dead_code)]\n\
             pub mod {name} {{\n    \
                 include!(concat!(env!(\"OUT_DIR\"), \"/model/{name}.rs\"));\n\
             }}\n"
        )
        .unwrap();
    }
    let path = out_dir.join("models.rs");
    fs::write(&path, buf).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Emit `$OUT_DIR/harness.rs`: one `#[test] fn <name>()` per test that
/// passed introspection. Each function is statically typed against the
/// introspected input/output shapes and dtypes; a rank or dtype
/// mismatch between this emission and the actual generated Model
/// produces a rustc error rather than a silent runtime bug.
fn emit_harness_rs(out_dir: &Path, harnessable: &BTreeMap<String, TestMeta>) {
    let mut buf = String::new();
    buf.push_str(
        "// GENERATED by build.rs — do not edit. One #[test] per\n\
         // harnessable entry from expectations.toml (status = pass).\n\n",
    );

    for (name, meta) in harnessable {
        emit_single_test(&mut buf, name, meta);
    }

    let path = out_dir.join("harness.rs");
    fs::write(&path, buf).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Emit one `#[test] fn <name>()` into `buf`.
fn emit_single_test(buf: &mut String, name: &str, meta: &TestMeta) {
    writeln!(buf, "#[test]").unwrap();
    writeln!(buf, "#[allow(non_snake_case)]").unwrap();
    writeln!(buf, "fn {name}() {{").unwrap();
    writeln!(buf, "    let device = Default::default();").unwrap();
    writeln!(
        buf,
        "    let model = generated::{name}::Model::<TestBackend>::new(&device);"
    )
    .unwrap();

    // ---- Load inputs ----
    let mut input_bindings: Vec<String> = Vec::new();
    for (idx, inp) in meta.inputs.iter().enumerate() {
        let binding = format!("input_{idx}");
        let rank = inp.rank;
        let kind = inp.dtype.tensor_kind_suffix();
        let variant = inp.dtype.values_variant();
        let onnx_name = inp.dtype.onnx_name();
        let dtype_tokens = inp.dtype.burn_dtype_tokens();

        writeln!(
            buf,
            "    let {binding}_ref = onnx_official_tests::pb_loader::load_tensor(\
             &vendor_file(\"{name}\", \"input_{idx}.pb\"))\n\
             \x20       .unwrap_or_else(|e| panic!(\"{name} input_{idx}: {{e}}\"));"
        )
        .unwrap();
        writeln!(
            buf,
            "    assert_eq!(\
             {binding}_ref.shape.len(), {rank}, \
             \"{name} input_{idx}: introspected rank {rank} but .pb has rank {{}}\", \
             {binding}_ref.shape.len());"
        )
        .unwrap();
        // Pass `(device, DType::X)` to `from_data` so the tensor's runtime dtype
        // is pinned to what the `.pb` source carries. A bare `from_data(data,
        // &device)` would convert to the backend's default int/float element,
        // which varies by backend (Flex=I32, NdArray=I64) and would silently
        // mutate the test's input dtype.
        writeln!(
            buf,
            "    let {binding}: burn::tensor::Tensor<TestBackend, {rank}{kind}> = match {binding}_ref.values {{\n\
             \x20       onnx_official_tests::pb_loader::TensorValues::{variant}(values) => {{\n\
             \x20           let data = burn::tensor::TensorData::new(values, {binding}_ref.shape);\n\
             \x20           burn::tensor::Tensor::<TestBackend, {rank}{kind}>::from_data(data, (&device, {dtype_tokens}))\n\
             \x20       }}\n\
             \x20       other => panic!(\"{name} input_{idx}: expected {onnx_name}, got {{}}\", other.dtype_name()),\n\
             \x20   }};"
        )
        .unwrap();
        input_bindings.push(binding);
    }

    // ---- Call forward ----
    let call_args = input_bindings.join(", ");
    if meta.outputs.len() == 1 {
        writeln!(buf, "    let output_0 = model.forward({call_args});").unwrap();
    } else {
        let destructure: Vec<String> = (0..meta.outputs.len())
            .map(|i| format!("output_{i}"))
            .collect();
        writeln!(
            buf,
            "    let ({}) = model.forward({call_args});",
            destructure.join(", ")
        )
        .unwrap();
    }

    // ---- Compare outputs ----
    //
    // We build `expected_<i>_data: TensorData` first and then compare
    // its `.shape` (a burn `Shape`) to `actual.shape` (also a `Shape`).
    // Comparing raw `Vec<usize>` from the loader to the `Shape` field
    // would be a type mismatch, and asking the generated harness to
    // call `.into()` is both noisy and order-dependent.
    for (idx, outp) in meta.outputs.iter().enumerate() {
        let binding = format!("output_{idx}");
        let variant = outp.dtype.values_variant();
        let onnx_name = outp.dtype.onnx_name();

        writeln!(
            buf,
            "    let expected_{idx}_ref = onnx_official_tests::pb_loader::load_tensor(\
             &vendor_file(\"{name}\", \"output_{idx}.pb\"))\n\
             \x20       .unwrap_or_else(|e| panic!(\"{name} output_{idx}: {{e}}\"));"
        )
        .unwrap();
        writeln!(
            buf,
            "    let expected_{idx}_data = match expected_{idx}_ref.values {{\n\
             \x20       onnx_official_tests::pb_loader::TensorValues::{variant}(values) => \
                         burn::tensor::TensorData::new(values, expected_{idx}_ref.shape),\n\
             \x20       other => panic!(\"{name} output_{idx}: expected {onnx_name}, got {{}}\", other.dtype_name()),\n\
             \x20   }};"
        )
        .unwrap();
        writeln!(buf, "    let actual_{idx}_data = {binding}.to_data();").unwrap();
        writeln!(
            buf,
            "    assert_eq!(\
             actual_{idx}_data.shape, expected_{idx}_data.shape, \
             \"{name} output_{idx} shape mismatch\");"
        )
        .unwrap();
        if outp.dtype.is_float() {
            let elem_ty = outp.dtype.assert_elem_type();
            writeln!(
                buf,
                "    actual_{idx}_data.assert_approx_eq::<{elem_ty}>\
                 (&expected_{idx}_data, burn::tensor::Tolerance::default());"
            )
            .unwrap();
        } else {
            // Strict dtype-aware comparison. Per CLAUDE.md, the generated
            // codegen must pin output dtypes explicitly rather than leak
            // the backend's default IntElem/BoolStore, so the actual and
            // expected dtypes should match without any normalization here.
            writeln!(
                buf,
                "    actual_{idx}_data.assert_eq(&expected_{idx}_data, true);"
            )
            .unwrap();
        }
    }

    writeln!(buf, "}}\n").unwrap();
}

/// Emit `$OUT_DIR/manifest.rs`: a sorted list of test names that the
/// drift check (in tests/test_mod.rs) compares against expectations.toml.
/// Two lists are emitted: `HARNESS_TESTS` (those with #[test] fns) and
/// `CODEGEN_ONLY_TESTS` (pass-listed but skipped from harness due to
/// introspection edge cases). Together they must equal the pass-set.
fn emit_manifest_rs(
    out_dir: &Path,
    harnessable: &BTreeMap<String, TestMeta>,
    codegen_only: &[String],
) {
    let mut buf = String::new();
    buf.push_str("// GENERATED by build.rs — do not edit.\n\n");

    buf.push_str("const HARNESS_TESTS: &[&str] = &[\n");
    for name in harnessable.keys() {
        writeln!(buf, "    \"{name}\",").unwrap();
    }
    buf.push_str("];\n\n");

    buf.push_str("const CODEGEN_ONLY_TESTS: &[&str] = &[\n");
    let mut sorted_co = codegen_only.to_vec();
    sorted_co.sort();
    for name in sorted_co {
        writeln!(buf, "    \"{name}\",").unwrap();
    }
    buf.push_str("];\n");

    let path = out_dir.join("manifest.rs");
    fs::write(&path, buf).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
