#!/usr/bin/env python3
"""Generate Rust test files for opset compliance testing.

Reads the manifest.txt and generates per-opset Rust test modules.
Each op test either expects success (snapshot) or failure (when min_opset > opset).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# Minimum opset each op supports in onnx-ir (from processor spec().min_opset).
# Used by gen_opset_compliance.py to split ops into passing/failing per opset.
MIN_OPSET = {
    # Arithmetic
    "Add": 1,
    "Sub": 1,
    "Mul": 1,
    "Div": 1,
    "Pow": 1,
    "Max": 1,
    "Min": 1,
    "Sum": 1,
    "Mean": 1,
    "Mod": 10,
    # Unary math
    "Abs": 1,
    "Neg": 1,
    "Ceil": 1,
    "Floor": 1,
    "Sqrt": 1,
    "Exp": 1,
    "Log": 1,
    "Reciprocal": 1,
    "Round": 11,
    "Sign": 9,
    "Erf": 9,
    # Trig
    "Sin": 7,
    "Cos": 7,
    "Tan": 7,
    "Sinh": 9,
    "Cosh": 9,
    "Tanh": 1,
    "Asin": 7,
    "Acos": 7,
    "Atan": 7,
    "Asinh": 9,
    "Acosh": 9,
    "Atanh": 9,
    # Activations
    "Relu": 1,
    "Sigmoid": 1,
    "Softplus": 1,
    "Softsign": 1,
    "Mish": 18,
    "Gelu": 20,
    "Celu": 12,
    "Elu": 1,
    "Selu": 1,
    "LeakyRelu": 1,
    "HardSigmoid": 1,
    "HardSwish": 14,
    "Hardmax": 1,
    "Softmax": 1,
    "LogSoftmax": 1,
    "PRelu": 1,
    "ThresholdedRelu": 10,
    "Swish": 24,
    # Comparison
    "Equal": 1,
    "Greater": 1,
    "Less": 1,
    "GreaterOrEqual": 12,
    "LessOrEqual": 12,
    # Logical
    "And": 1,
    "Or": 1,
    "Xor": 1,
    "Not": 1,
    # Bitwise
    "BitwiseAnd": 18,
    "BitwiseOr": 18,
    "BitwiseXor": 18,
    "BitwiseNot": 18,
    "BitShift": 11,
    # Reduction
    "ReduceMax": 1,
    "ReduceMin": 1,
    "ReduceMean": 1,
    "ReduceSum": 1,
    "ReduceProd": 1,
    "ReduceL1": 1,
    "ReduceL2": 1,
    "ReduceLogSum": 1,
    "ReduceLogSumExp": 1,
    "ReduceSumSquare": 1,
    "ArgMax": 1,
    "ArgMin": 1,
    # Shape/manipulation
    "Reshape": 1,
    "Transpose": 1,
    "Flatten": 1,
    "Squeeze": 1,
    "Unsqueeze": 1,
    "Concat": 1,
    "Split": 1,
    "Gather": 1,
    "GatherElements": 11,
    "GatherND": 11,
    "Slice": 1,
    "Tile": 1,
    "Expand": 8,
    "Pad": 1,
    "Clip": 1,
    "Shape": 1,
    "Size": 1,
    "DepthToSpace": 1,
    "SpaceToDepth": 1,
    "ScatterElements": 11,
    "ScatterND": 11,
    # Matrix
    "MatMul": 1,
    "Gemm": 1,
    "MatMulInteger": 10,
    # Conv
    "Conv": 1,
    "ConvTranspose": 1,
    # Pooling
    "AveragePool": 1,
    "MaxPool": 1,
    "GlobalAveragePool": 1,
    # Normalization
    "BatchNormalization": 1,
    "InstanceNormalization": 1,
    "LayerNormalization": 17,
    "GroupNormalization": 18,
    # Utility
    "Dropout": 1,
    "Identity": 1,
    "Cast": 1,
    "Where": 9,
    "NonZero": 9,
    "Constant": 1,
    "ConstantOfShape": 9,
    "OneHot": 9,
    "TopK": 1,
    "IsInf": 10,
    "IsNaN": 9,
    "Trilu": 14,
    "CumSum": 11,
    "EyeLike": 9,
    "Range": 11,
    "Resize": 10,
    "GridSample": 16,
    # Random
    "RandomNormal": 1,
    "RandomUniform": 1,
    "RandomNormalLike": 1,
    "RandomUniformLike": 1,
    "Bernoulli": 15,
    # RNN
    "LSTM": 1,
    "GRU": 1,
    # Control flow
    "If": 1,
    # DeformConv
    "DeformConv": 19,
}
MANIFEST_PATH = (
    REPO_ROOT
    / "crates"
    / "onnx-ir"
    / "tests"
    / "fixtures"
    / "opset_compliance"
    / "manifest.txt"
)
TEST_DIR = REPO_ROOT / "crates" / "onnx-ir" / "tests" / "opset_compliance"

def snake_case(name: str) -> str:
    """Convert CamelCase/PascalCase to snake_case."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0 and not name[i - 1].isupper():
            result.append("_")
        elif (
            c.isupper()
            and i > 0
            and name[i - 1].isupper()
            and i + 1 < len(name)
            and name[i + 1].islower()
        ):
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def test_fn_name(op_name: str) -> str:
    """Generate Rust test function name from ONNX op name."""
    s = snake_case(op_name)
    # Handle Rust reserved words
    RUST_KEYWORDS = {
        "and", "as", "break", "const", "continue", "crate", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop",
        "match", "mod", "move", "mut", "not", "or", "pub", "ref", "return",
        "self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "xor",
    }
    if s in RUST_KEYWORDS:
        return f"{s}_op"
    return s


# Map from ONNX spec name to the NodeType name used by the parser.
# The parser renames nodes as: "{NodeType}{counter}".to_lowercase()
# For dimensional variants, the parser maps ONNX op to burn's NodeType
# based on input tensor rank.
ONNX_TO_NODE_TYPE = {
    "Conv": "Conv2d",  # 4D input -> Conv2d
    "ConvTranspose": "ConvTranspose2d",
    "AveragePool": "AveragePool2d",
    "MaxPool": "MaxPool2d",
    "LSTM": "Lstm",
    "GRU": "Gru",
}


def node_name_prefix(op_name: str) -> str:
    """Get the node name prefix the parser will assign for this op."""
    node_type = ONNX_TO_NODE_TYPE.get(op_name, op_name)
    return node_type.lower()


# Ops eliminated during post-processing (no-op elimination)
NOOP_OPS = {"Dropout", "Identity"}


def generate_opset_file(opset: int, passing: list[str], failing: list[str]) -> str:
    """Generate Rust test module for one opset version."""
    # Check if we have any non-noop passing ops (need the fixture)
    has_snapshot_tests = any(op not in NOOP_OPS for op in passing)

    lines = [
        f"//! Opset {opset} compliance tests.",
        f"//!",
        f"//! Auto-generated by gen_rust_tests.py. Edit the generator, not this file.",
        "",
        "use super::helpers::*;",
    ]

    if has_snapshot_tests:
        lines.append("use rstest::*;")

    lines.append("")

    if has_snapshot_tests:
        lines.append("#[fixture]")
        lines.append("#[once]")
        lines.append("fn graph() -> OnnxGraph {")
        lines.append(f'    load_model("opset_{opset:02d}.onnx")')
        lines.append("}")
        lines.append("")

    # Passing ops: each gets a snapshot test using the shared fixture
    for op in sorted(passing):
        if op in NOOP_OPS:
            # These ops are eliminated during post-processing
            fn_name = test_fn_name(op)
            lines.append(f"/// {op} is eliminated during post-processing (no-op).")
            lines.append(f"/// Verify the model parses without error.")
            lines.append("#[test]")
            lines.append(f"fn {fn_name}() {{")
            lines.append(f'    let _graph = load_model("opset_{opset:02d}.onnx");')
            lines.append(f"}}")
            lines.append("")
        else:
            fn_name = test_fn_name(op)
            prefix = node_name_prefix(op)
            lines.append("#[rstest]")
            lines.append(f"fn {fn_name}(graph: &OnnxGraph) {{")
            lines.append(f'    let node = find_node(graph, "{prefix}");')
            lines.append(f'    insta::assert_snapshot!(format!("{{node}}"), @r"");')
            lines.append(f"}}")
            lines.append("")

    # Failing ops: the unsupported model should fail to parse
    if failing:
        ops_list = ", ".join(failing)
        lines.append(f"/// Ops that require min_opset > {opset}: {ops_list}")
        lines.append("#[test]")
        lines.append(f"fn unsupported_ops_fail() {{")
        lines.append(f'    let result = load_model_result("opset_{opset:02d}_unsupported.onnx");')
        lines.append(f"    assert!(result.is_err(), \"expected parse failure for unsupported ops at opset {opset}\");")
        lines.append(f"}}")
        lines.append("")

    return "\n".join(lines)


def main():
    manifest = MANIFEST_PATH.read_text().strip()
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    opset_modules = []

    for line in manifest.splitlines():
        if not line.strip():
            continue
        opset_str, rest = line.split(":", 1)
        opset = int(opset_str)

        # New manifest format: opset:passing_ops|failing_ops
        if "|" in rest:
            pass_str, fail_str = rest.split("|", 1)
        else:
            pass_str, fail_str = rest, ""

        passing = [o.strip() for o in pass_str.split(",") if o.strip()]
        failing = [o.strip() for o in fail_str.split(",") if o.strip()]

        if not passing and not failing:
            continue

        mod_name = f"opset_{opset:02d}"
        content = generate_opset_file(opset, passing, failing)

        filepath = TEST_DIR / f"{mod_name}.rs"
        filepath.write_text(content + "\n")
        opset_modules.append(mod_name)
        print(f"  Generated {filepath.name} ({len(passing)} passing, {len(failing)} failing)")

    # Write the main entry point
    entry = TEST_DIR.parent / "opset_compliance.rs"
    entry_lines = [
        "//! Opset compliance tests for onnx-ir.",
        "//!",
        "//! Tests that every supported ONNX op can be parsed at every opset version",
        "//! where the op's spec changed.",
        "//!",
        "//! Auto-generated by gen_rust_tests.py. Edit the generator, not this file.",
        "",
        "mod opset_compliance {",
        "    mod helpers;",
    ]
    for mod_name in sorted(opset_modules):
        entry_lines.append(f"    mod {mod_name};")
    entry_lines.append("}")
    entry_lines.append("")

    entry.write_text("\n".join(entry_lines))
    print(f"\n  Generated {entry.name} with {len(opset_modules)} opset modules")


if __name__ == "__main__":
    main()
