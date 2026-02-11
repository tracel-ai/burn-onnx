#!/usr/bin/env python3
"""Generate Rust test files for opset compliance testing.

Reads the manifest.txt and generates per-opset Rust test modules.
Each op test either expects success (snapshot) or failure (when min_opset > opset).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
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

# min_opset for each ONNX spec name as registered in burn-onnx processors.
# For dimensional variants (Conv->Conv2d, etc.), use the 2d variant's min_opset.
MIN_OPSET = {
    "Abs": 6,
    "Acos": 7,
    "Acosh": 9,
    "Add": 7,
    "And": 1,
    "ArgMax": 11,
    "ArgMin": 11,
    "Asin": 7,
    "Asinh": 9,
    "Atan": 7,
    "Atanh": 9,
    "AveragePool": 11,
    "BatchNormalization": 9,
    "Bernoulli": 15,
    "BitShift": 11,
    "BitwiseAnd": 18,
    "BitwiseNot": 18,
    "BitwiseOr": 18,
    "BitwiseXor": 18,
    "Cast": 1,
    "Ceil": 6,
    "Celu": 12,
    "Clip": 6,
    "Concat": 4,
    "Constant": 1,
    "ConstantOfShape": 9,
    "Conv": 1,
    "ConvTranspose": 1,
    "Cos": 7,
    "Cosh": 9,
    "CumSum": 11,
    "DeformConv": 19,
    "DepthToSpace": 1,
    "Div": 7,
    "Dropout": 1,
    "Elu": 1,
    "Equal": 7,
    "Erf": 9,
    "Exp": 6,
    "Expand": 8,
    "EyeLike": 9,
    "Flatten": 1,
    "Floor": 6,
    "GRU": 1,
    "Gather": 1,
    "GatherElements": 11,
    "GatherND": 11,
    "Gelu": 20,
    "Gemm": 11,
    "GlobalAveragePool": 1,
    "Greater": 7,
    "GreaterOrEqual": 12,
    "GridSample": 16,
    "GroupNormalization": 18,
    "HardSigmoid": 6,
    "HardSwish": 14,
    "Hardmax": 13,
    "Identity": 1,
    "If": 1,
    "InstanceNormalization": 6,
    "IsInf": 10,
    "IsNaN": 9,
    "LSTM": 1,
    "LayerNormalization": 17,
    "LeakyRelu": 6,
    "Less": 7,
    "LessOrEqual": 12,
    "Log": 6,
    "LogSoftmax": 13,
    "MatMul": 1,
    "MatMulInteger": 10,
    "Max": 1,
    "MaxPool": 1,
    "Mean": 8,
    "Min": 1,
    "Mish": 18,
    "Mod": 10,
    "Mul": 7,
    "Neg": 6,
    "NonZero": 9,
    "Not": 1,
    "OneHot": 9,
    "Or": 1,
    "PRelu": 6,
    "Pad": 11,
    "Pow": 1,
    "RandomNormal": 1,
    "RandomNormalLike": 1,
    "RandomUniform": 1,
    "RandomUniformLike": 1,
    "Range": 11,
    "Reciprocal": 6,
    "ReduceL1": 11,
    "ReduceL2": 11,
    "ReduceLogSum": 11,
    "ReduceLogSumExp": 11,
    "ReduceMax": 11,
    "ReduceMean": 11,
    "ReduceMin": 11,
    "ReduceProd": 11,
    "ReduceSum": 11,
    "ReduceSumSquare": 11,
    "Relu": 6,
    "Reshape": 5,
    "Resize": 11,
    "Round": 11,
    "ScatterElements": 11,
    "ScatterND": 11,
    "Selu": 1,
    "Shape": 1,
    "Sigmoid": 6,
    "Sign": 9,
    "Sin": 7,
    "Sinh": 9,
    "Size": 1,
    "Slice": 10,
    "Softmax": 13,
    "Softplus": 1,
    "Softsign": 1,
    "SpaceToDepth": 1,
    "Split": 11,
    "Sqrt": 6,
    "Squeeze": 13,
    "Sub": 7,
    "Sum": 8,
    "Swish": 24,
    "Tan": 7,
    "Tanh": 6,
    "ThresholdedRelu": 10,
    "Tile": 6,
    "TopK": 10,
    "Transpose": 1,
    "Trilu": 14,
    "Unsqueeze": 13,
    "Where": 9,
    "Xor": 1,
}


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
    lines = [
        f"//! Opset {opset} compliance tests.",
        f"//!",
        f"//! Auto-generated by gen_rust_tests.py. Edit the generator, not this file.",
        "",
        "use super::helpers::*;",
        "",
    ]

    # Passing ops: each gets a snapshot test using the passing-only model
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
            lines.append("#[test]")
            lines.append(f"fn {fn_name}() {{")
            lines.append(f'    let graph = load_model("opset_{opset:02d}.onnx");')
            lines.append(f'    let node = find_node(&graph, "{prefix}");')
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
