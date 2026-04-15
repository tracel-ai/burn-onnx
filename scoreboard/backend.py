"""ONNX backend wrapper for burn-onnx (Rust codegen inference engine)."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt

# Path to the runner template project (pre-compiled burn deps).
# In Docker, this is /root/runner. Locally, relative to this file.
RUNNER_DIR = os.environ.get(
    "BURN_RUNNER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner"),
)

# Path to onnx2burn binary.
ONNX2BURN = os.environ.get("ONNX2BURN", "onnx2burn")

# Cargo build timeout (seconds). First build compiles burn deps (~5 min).
# Incremental builds are much faster (~10-30s).
CARGO_TIMEOUT = int(os.environ.get("CARGO_TIMEOUT", "120"))

# Inference timeout per model (seconds).
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "60"))


# -- ONNX elem_type constants --
ONNX_FLOAT = 1
ONNX_DOUBLE = 11
ONNX_FLOAT16 = 10
ONNX_INT8 = 3
ONNX_INT16 = 5
ONNX_INT32 = 6
ONNX_INT64 = 7
ONNX_UINT8 = 2
ONNX_BOOL = 9

ONNX_DTYPE_TO_NUMPY = {
    ONNX_FLOAT: np.float32,
    ONNX_DOUBLE: np.float64,
    ONNX_FLOAT16: np.float16,
    ONNX_INT8: np.int8,
    ONNX_INT16: np.int16,
    ONNX_INT32: np.int32,
    ONNX_INT64: np.int64,
    ONNX_UINT8: np.uint8,
    ONNX_BOOL: np.bool_,
}

NUMPY_DTYPE_TO_STR = {
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("float16"): "float16",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("bool"): "bool",
}

# Rust DType string from TensorData.dtype debug output -> numpy dtype.
#
# Intentional omission: bf16. bf16 and fp16 are both 2 bytes wide but have
# different bit layouts (8 exponent bits vs 5), so reading raw bf16 bytes
# via `np.frombuffer(..., dtype=np.float16)` would be a silent
# reinterpretation producing garbage values, not a lossy conversion. If a
# model produces bf16 outputs, we want `_resolve_rust_dtype` to explicitly
# raise `BackendIsNotSupposedToImplementIt` so the scoreboard skips the
# model rather than submits wrong results upstream. Support bf16 properly
# by either (a) teaching the Rust runner to cast bf16 outputs to f32
# before writing, or (b) reading them as u16 and unpacking manually.
RUST_DTYPE_TO_NUMPY = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "flex32": np.float32,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "bool": np.bool_,
}


def _parse_forward_signature(rs_source):
    """Parse the forward() method signature from generated Rust source.

    Returns (params, return_type) where:
      params: list of (name, rust_type_str) excluding &self
      return_type: rust_type_str (may be a tuple like "(Tensor<B, 2>, Tensor<B, 1, Int>)")
    """
    # Match across potential line breaks in the signature
    pattern = re.compile(
        r"pub\s+fn\s+forward\s*\(\s*&self\s*,?\s*(.*?)\)\s*->\s*(.+?)\s*\{",
        re.DOTALL,
    )
    m = pattern.search(rs_source)
    if not m:
        raise BackendIsNotSupposedToImplementIt(
            "Could not parse forward() signature from generated code"
        )

    params_str = m.group(1).strip()
    return_str = m.group(2).strip()

    params = []
    if params_str:
        # Split on commas, but not commas inside angle brackets
        depth = 0
        current = []
        for ch in params_str:
            if ch in "<(":
                depth += 1
            elif ch in ">)":
                depth -= 1
            if ch == "," and depth == 0:
                params.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            params.append("".join(current).strip())

        # Parse "name: Type" pairs
        parsed = []
        for p in params:
            p = p.strip()
            if not p:
                continue
            name, ty = p.split(":", 1)
            parsed.append((name.strip(), ty.strip()))
        params = parsed

    return params, return_str


def _parse_return_types(return_str):
    """Parse return type string into a list of individual types.

    "Tensor<B, 4>" -> ["Tensor<B, 4>"]
    "(Tensor<B, 2, Int>, Tensor<B, 1>)" -> ["Tensor<B, 2, Int>", "Tensor<B, 1>"]
    """
    s = return_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    # Split on top-level commas
    types = []
    depth = 0
    current = []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            types.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        types.append("".join(current).strip())

    return types


def _rust_type_category(ty):
    """Classify a Rust type string.

    Returns one of: "float_tensor", "int_tensor", "bool_tensor",
                    "scalar_f32", "scalar_f64", "scalar_i64", "scalar_i32",
                    "scalar_bool", "unknown"
    """
    if ty.startswith("Tensor<"):
        if "Bool" in ty:
            return "bool_tensor"
        elif "Int" in ty:
            return "int_tensor"
        else:
            return "float_tensor"
    scalar_map = {
        "f32": "scalar_f32",
        "f64": "scalar_f64",
        "i64": "scalar_i64",
        "i32": "scalar_i32",
        "bool": "scalar_bool",
    }
    return scalar_map.get(ty, "unknown")


def _tensor_rank(ty):
    """Extract rank N from "Tensor<B, N>" or "Tensor<B, N, Int>"."""
    m = re.search(r"Tensor<\s*B\s*,\s*(\d+)", ty)
    if m:
        return int(m.group(1))
    return None


def _gen_read_input(idx, name, rust_type):
    """Generate Rust code to read one input from the manifest.

    The emitted `Tensor::from_data` calls all pin the tensor's runtime dtype
    via `(&device, burn::tensor::DType::X)`. Leaving the second argument as
    a bare `&device` would let burn's `from_data` convert the TensorData to
    the backend's default Int/Float element type, which varies across
    backends and can silently truncate an int64 source tensor (for example
    when the backend's default IntElem is narrower than i64). That also
    violates the project's "explicit dtypes in generated code" rule (see
    .claude/CLAUDE.md). The numpy inputs are cast to the matching precision
    on the Python side before serialization (see `_CATEGORY_TO_NUMPY`), so
    these dtype constants are the correct runtime width for each branch.
    """
    cat = _rust_type_category(rust_type)
    rank = _tensor_rank(rust_type)

    if cat == "float_tensor":
        return (
            f"    let {name} = {{\n"
            f"        let info = &manifest.inputs[{idx}];\n"
            f"        let bytes = std::fs::read(&info.file).expect(\"read input\");\n"
            f"        let values: Vec<f32> = bytes.chunks_exact(4)\n"
            f"            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();\n"
            f"        Tensor::<B, {rank}>::from_data(\n"
            f"            TensorData::new(values, info.shape.clone()),\n"
            f"            (&device, burn::tensor::DType::F32),\n"
            f"        )\n"
            f"    }};\n"
        )
    elif cat == "int_tensor":
        return (
            f"    let {name} = {{\n"
            f"        let info = &manifest.inputs[{idx}];\n"
            f"        let bytes = std::fs::read(&info.file).expect(\"read input\");\n"
            f"        let values: Vec<i64> = bytes.chunks_exact(8)\n"
            f"            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();\n"
            f"        Tensor::<B, {rank}, Int>::from_data(\n"
            f"            TensorData::new(values, info.shape.clone()),\n"
            f"            (&device, burn::tensor::DType::I64),\n"
            f"        )\n"
            f"    }};\n"
        )
    elif cat == "bool_tensor":
        return (
            f"    let {name} = {{\n"
            f"        let info = &manifest.inputs[{idx}];\n"
            f"        let bytes = std::fs::read(&info.file).expect(\"read input\");\n"
            f"        let values: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();\n"
            f"        Tensor::<B, {rank}, Bool>::from_data(\n"
            f"            TensorData::new(values, info.shape.clone()),\n"
            f"            (&device, burn::tensor::DType::Bool(burn::tensor::BoolStore::Native)),\n"
            f"        )\n"
            f"    }};\n"
        )
    elif cat in ("scalar_i64", "scalar_i32", "scalar_f32", "scalar_f64"):
        # Fixed-size native scalar: validate length then decode.
        scalar_info = {
            "scalar_i64": ("i64", 8),
            "scalar_i32": ("i32", 4),
            "scalar_f32": ("f32", 4),
            "scalar_f64": ("f64", 8),
        }[cat]
        rust_ty, nbytes = scalar_info
        return (
            f"    let {name} = {{\n"
            f"        let info = &manifest.inputs[{idx}];\n"
            f"        let bytes = std::fs::read(&info.file).expect(\"read input\");\n"
            f"        assert_eq!(bytes.len(), {nbytes}, \"expected {nbytes} bytes for {rust_ty}\");\n"
            f"        {rust_ty}::from_le_bytes(bytes[..{nbytes}].try_into().unwrap())\n"
            f"    }};\n"
        )
    elif cat == "scalar_bool":
        return (
            f"    let {name} = {{\n"
            f"        let info = &manifest.inputs[{idx}];\n"
            f"        let bytes = std::fs::read(&info.file).expect(\"read input\");\n"
            f"        assert!(!bytes.is_empty(), \"expected at least 1 byte for bool\");\n"
            f"        bytes[0] != 0\n"
            f"    }};\n"
        )
    else:
        raise BackendIsNotSupposedToImplementIt(
            f"Unsupported input type: {rust_type}"
        )


def _gen_write_output(idx, var_expr, rust_type):
    """Generate Rust code to write one output."""
    cat = _rust_type_category(rust_type)

    if cat in ("float_tensor", "int_tensor", "bool_tensor"):
        return (
            f"    {{\n"
            f"        let data = {var_expr}.to_data();\n"
            f"        let shape: Vec<usize> = data.shape.iter().copied().collect();\n"
            f"        let dtype = format!(\"{{:?}}\", data.dtype).to_lowercase();\n"
            f"        let path = format!(\"{{output_dir}}/output_{idx}.bin\");\n"
            f"        std::fs::write(&path, data.as_bytes()).expect(\"write output\");\n"
            f"        out_infos.push(serde_json::json!({{\"file\": path, \"shape\": shape, \"dtype\": dtype}}));\n"
            f"    }}\n"
        )
    elif cat in ("scalar_f32", "scalar_f64", "scalar_i64", "scalar_i32"):
        dtype_str = {
            "scalar_f32": "float32",
            "scalar_f64": "float64",
            "scalar_i64": "int64",
            "scalar_i32": "int32",
        }[cat]
        return (
            f"    {{\n"
            f"        let path = format!(\"{{output_dir}}/output_{idx}.bin\");\n"
            f"        std::fs::write(&path, {var_expr}.to_le_bytes()).expect(\"write output\");\n"
            f"        out_infos.push(serde_json::json!({{\"file\": path, \"shape\": [], \"dtype\": \"{dtype_str}\"}}));\n"
            f"    }}\n"
        )
    elif cat == "scalar_bool":
        # bool has no to_le_bytes(); write a single 0/1 byte.
        return (
            f"    {{\n"
            f"        let path = format!(\"{{output_dir}}/output_{idx}.bin\");\n"
            f"        std::fs::write(&path, [u8::from({var_expr})]).expect(\"write output\");\n"
            f"        out_infos.push(serde_json::json!({{\"file\": path, \"shape\": [], \"dtype\": \"bool\"}}));\n"
            f"    }}\n"
        )
    else:
        raise BackendIsNotSupposedToImplementIt(
            f"Unsupported output type: {rust_type}"
        )


def _generate_main_rs(rs_source, bpk_path):
    """Generate a main.rs harness from the parsed forward() signature."""
    params, return_str = _parse_forward_signature(rs_source)
    return_types = _parse_return_types(return_str)
    is_tuple = len(return_types) > 1

    lines = [
        "#[path = \"model.rs\"]",
        "mod model;",
        "",
        "use burn::prelude::*;",
        "use burn::tensor::TensorData;",
        "use burn::backend::Flex;",
        "use serde::Deserialize;",
        "",
        "type B = Flex;",
        "",
        "#[derive(Deserialize)]",
        "struct Manifest {",
        "    inputs: Vec<InputInfo>,",
        "    bpk_path: String,",
        "}",
        "",
        "#[derive(Deserialize)]",
        "struct InputInfo {",
        "    file: String,",
        "    shape: Vec<usize>,",
        "    #[allow(dead_code)]",
        "    dtype: String,",
        "}",
        "",
        "fn main() {",
        "    let args: Vec<String> = std::env::args().collect();",
        "    let manifest_path = &args[1];",
        "    let output_dir = &args[2];",
        "",
        "    let manifest: Manifest = serde_json::from_str(",
        "        &std::fs::read_to_string(manifest_path).expect(\"read manifest\")",
        "    ).expect(\"parse manifest\");",
        "",
        "    let device = <B as burn::tensor::backend::Backend>::Device::default();",
        "    let model = model::Model::<B>::from_file(&manifest.bpk_path, &device);",
        "",
    ]

    # Generate input reads
    for idx, (name, rust_type) in enumerate(params):
        lines.append(_gen_read_input(idx, name, rust_type))

    # Generate forward call
    param_names = ", ".join(name for name, _ in params)

    if is_tuple:
        var_names = ", ".join(f"out_{i}" for i in range(len(return_types)))
        lines.append(f"    let ({var_names}) = model.forward({param_names});")
    else:
        lines.append(f"    let out_0 = model.forward({param_names});")

    lines.append("")
    lines.append("    let mut out_infos: Vec<serde_json::Value> = Vec::new();")

    # Generate output writes
    for idx, rt in enumerate(return_types):
        lines.append(_gen_write_output(idx, f"out_{idx}", rt))

    # Write output manifest
    lines.append(
        '    let manifest = serde_json::json!({"outputs": out_infos});'
    )
    lines.append(
        '    std::fs::write(format!("{output_dir}/manifest.json"), manifest.to_string()).expect("write manifest");'
    )
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# Numpy dtype expected by the Rust runner for each input category.
# The generated main.rs reads fixed types (f32 for float tensors, i64 for int
# tensors, etc.), so numpy inputs must be cast to match before serialization.
_CATEGORY_TO_NUMPY = {
    "float_tensor": np.float32,
    "int_tensor": np.int64,
    "bool_tensor": np.bool_,
    "scalar_f32": np.float32,
    "scalar_f64": np.float64,
    "scalar_i64": np.int64,
    "scalar_i32": np.int32,
    "scalar_bool": np.bool_,
}


def _serialize_input(arr, path):
    """Write a numpy array as raw little-endian bytes."""
    arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _write_input_manifest(inputs, input_types, input_dir, bpk_path, manifest_path):
    """Write the input manifest JSON, casting inputs to the runner's expected dtype.

    `input_types` is a list of parsed Rust type strings from the forward()
    signature, one per input. Each numpy array is cast to the dtype the runner
    expects (e.g. float tensors -> f32) to avoid byte-level corruption.
    """
    input_infos = []
    for i, (arr, rust_type) in enumerate(zip(inputs, input_types)):
        arr = np.asarray(arr)
        cat = _rust_type_category(rust_type)
        target = _CATEGORY_TO_NUMPY.get(cat)
        if target is None:
            raise BackendIsNotSupposedToImplementIt(
                f"Unsupported input type for serialization: {rust_type}"
            )
        if arr.dtype != np.dtype(target):
            # A lossy cast (e.g. uint64 -> int64 with out-of-range values)
            # means the runner can't represent this input faithfully; surface
            # it as a skip rather than producing garbage inference.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    arr = arr.astype(target)
                except RuntimeWarning as w:
                    raise BackendIsNotSupposedToImplementIt(
                        f"Lossy input cast {arr.dtype} -> {np.dtype(target)}: {w}"
                    )
        file_path = os.path.join(input_dir, f"input_{i}.bin")
        _serialize_input(arr, file_path)
        dtype_str = NUMPY_DTYPE_TO_STR.get(arr.dtype, str(arr.dtype))
        input_infos.append({
            "file": file_path,
            "shape": list(arr.shape),
            "dtype": dtype_str,
        })

    manifest = {"inputs": input_infos, "bpk_path": bpk_path}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def _resolve_rust_dtype(dtype_str):
    """Map Burn's DType debug string to numpy dtype.

    Burn formats DType as e.g. "F32", "I64", "Bool(native)".
    After .to_lowercase() in the runner: "f32", "i64", "bool(native)".

    Raises `BackendIsNotSupposedToImplementIt` for dtypes the scoreboard
    cannot safely read back — bf16 (byte layout mismatch with fp16) and
    any future dtype name we haven't mapped. A silent fallback to f32
    would cause the scoreboard to submit wrong-answer results upstream
    when a model produces an unmapped dtype.
    """
    # Strip parenthesized qualifiers: "bool(native)" -> "bool"
    base = dtype_str.split("(")[0].strip()
    numpy_dtype = RUST_DTYPE_TO_NUMPY.get(base)
    if numpy_dtype is None:
        raise BackendIsNotSupposedToImplementIt(
            f"Output dtype '{dtype_str}' is not supported by the burn-onnx "
            f"scoreboard runner. Mapped dtypes: {sorted(RUST_DTYPE_TO_NUMPY)}. "
            f"Extend RUST_DTYPE_TO_NUMPY to add support."
        )
    return numpy_dtype


def _read_output_manifest(output_dir):
    """Read output manifest and return list of numpy arrays."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    outputs = []
    for info in manifest["outputs"]:
        dtype = _resolve_rust_dtype(info["dtype"])
        with open(info["file"], "rb") as f:
            raw = np.frombuffer(f.read(), dtype=dtype)
        shape = info["shape"]
        if shape:
            raw = raw.reshape(shape)
        outputs.append(raw)

    return outputs


class BurnOnnxBackendRep(BackendRep):
    """Holds a compiled burn-onnx model binary ready for inference."""

    def __init__(self, binary_path, bpk_path, work_dir, input_types):
        self._binary = binary_path
        self._bpk = bpk_path
        self._work_dir = work_dir
        self._input_types = input_types

    def run(self, inputs, **kwargs):
        input_dir = os.path.join(self._work_dir, "inputs")
        output_dir = os.path.join(self._work_dir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        manifest_path = os.path.join(self._work_dir, "input_manifest.json")
        _write_input_manifest(
            inputs, self._input_types, input_dir, self._bpk, manifest_path
        )

        try:
            result = subprocess.run(
                [self._binary, manifest_path, output_dir],
                capture_output=True,
                timeout=INFERENCE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise BackendIsNotSupposedToImplementIt("Inference timed out")

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise BackendIsNotSupposedToImplementIt(
                f"Inference failed (exit {result.returncode}): {stderr[:500]}"
            )

        try:
            return _read_output_manifest(output_dir)
        finally:
            # Clean up per-run input/output files so the work dir can be
            # deleted cleanly by __del__.
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    def __del__(self):
        # Best-effort cleanup: the test harness creates thousands of reps.
        work_dir = getattr(self, "_work_dir", None)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


class BurnOnnxBackend(Backend):
    """ONNX backend implementation using burn-onnx codegen."""

    @classmethod
    def is_compatible(cls, model, device="CPU", **kwargs):
        return True

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        work_dir = tempfile.mkdtemp(prefix="burn_onnx_")
        try:
            return cls._prepare_impl(model, work_dir)
        except BackendIsNotSupposedToImplementIt:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise BackendIsNotSupposedToImplementIt(
                f"Model preparation failed: {e}"
            )

    @classmethod
    def _prepare_impl(cls, model, work_dir):
        # 1. Save ONNX model
        onnx_path = os.path.join(work_dir, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(model.SerializeToString())

        # 2. Run onnx2burn codegen
        gen_dir = os.path.join(work_dir, "generated")
        os.makedirs(gen_dir)
        result = subprocess.run(
            [ONNX2BURN, onnx_path, gen_dir, "--no-development", "--no-simplify"],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise BackendIsNotSupposedToImplementIt(
                f"onnx2burn failed: {stderr[:500]}"
            )

        # 3. Find generated files (sorted for determinism; fail fast on
        # unexpected multiplicity)
        rs_files = sorted(f for f in os.listdir(gen_dir) if f.endswith(".rs"))
        bpk_files = sorted(f for f in os.listdir(gen_dir) if f.endswith(".bpk"))
        if len(rs_files) != 1:
            raise BackendIsNotSupposedToImplementIt(
                f"Expected exactly one .rs file, found {len(rs_files)}: {rs_files}"
            )
        if len(bpk_files) != 1:
            raise BackendIsNotSupposedToImplementIt(
                f"Expected exactly one .bpk file, found {len(bpk_files)}: {bpk_files}"
            )

        rs_path = os.path.join(gen_dir, rs_files[0])
        bpk_path = os.path.join(gen_dir, bpk_files[0])

        # 4. Read generated source, parse input types, and create main.rs
        with open(rs_path) as f:
            rs_source = f.read()

        params, _ = _parse_forward_signature(rs_source)
        input_types = [ty for _, ty in params]

        main_rs = _generate_main_rs(rs_source, bpk_path)

        # 5. Copy to runner project
        runner_src = os.path.join(RUNNER_DIR, "src")
        shutil.copy2(rs_path, os.path.join(runner_src, "model.rs"))
        with open(os.path.join(runner_src, "main.rs"), "w") as f:
            f.write(main_rs)

        # 6. Compile (incremental debug build, only model.rs + main.rs changed)
        result = subprocess.run(
            ["cargo", "build"],
            cwd=RUNNER_DIR,
            capture_output=True,
            timeout=CARGO_TIMEOUT,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise BackendIsNotSupposedToImplementIt(
                f"Cargo build failed: {stderr[:1000]}"
            )

        binary = os.path.join(RUNNER_DIR, "target", "debug", "burn-onnx-runner")
        if not os.path.exists(binary):
            raise BackendIsNotSupposedToImplementIt("Binary not found after build")

        return BurnOnnxBackendRep(binary, bpk_path, work_dir, input_types)

    @classmethod
    def run_model(cls, model, inputs, device="CPU", **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs)

    @classmethod
    def supports_device(cls, device):
        return device == "CPU"


prepare = BurnOnnxBackend.prepare
run_model = BurnOnnxBackend.run_model
supports_device = BurnOnnxBackend.supports_device
