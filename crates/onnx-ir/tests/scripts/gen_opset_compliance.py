#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.17.0",
#   "numpy==1.26.4",
# ]
# ///

"""Generate ONNX models for opset compliance testing.

For each opset version where at least one supported op changed, generates
one .onnx file containing all ops that have a spec change at that version.
Each op is an independent subgraph with its own inputs and outputs.

Usage:
    python crates/onnx-ir/tests/scripts/gen_opset_compliance.py
"""

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_DIR = REPO_ROOT / "onnx-spec" / "ops"
FIXTURE_DIR = (
    REPO_ROOT / "crates" / "onnx-ir" / "tests" / "fixtures" / "opset_compliance"
)

# ---------------------------------------------------------------------------
# Mapping from ONNX spec names to registered ops in burn-onnx registry.
# Some burn-onnx ops are dimensional variants (Conv1d, Conv2d, etc.) that
# come from a single ONNX op (Conv). We use ONNX spec names here.
#
# Ops that don't appear in this set are not registered in registry.rs and
# thus not tested.
# ---------------------------------------------------------------------------

# Map: ONNX spec name -> list of (burn_name, generator_key) tuples.
# For most ops, burn_name == spec_name. Dimensional variants are special.
SUPPORTED_OPS = {
    # Arithmetic
    "Add": "binary_float",
    "Sub": "binary_float",
    "Mul": "binary_float",
    "Div": "binary_float",
    "Pow": "binary_float",
    "Max": "variadic_float",
    "Min": "variadic_float",
    "Sum": "variadic_float",
    "Mean": "variadic_float",
    "Mod": "binary_float",
    # Unary math
    "Abs": "unary_float",
    "Neg": "unary_float",
    "Ceil": "unary_float",
    "Floor": "unary_float",
    "Sqrt": "unary_float",
    "Exp": "unary_float",
    "Log": "unary_float",
    "Reciprocal": "unary_float",
    "Round": "unary_float",
    "Sign": "unary_float",
    "Erf": "unary_float",
    # Trig
    "Sin": "unary_float",
    "Cos": "unary_float",
    "Tan": "unary_float",
    "Sinh": "unary_float",
    "Cosh": "unary_float",
    "Tanh": "unary_float",
    "Asin": "unary_float",
    "Acos": "unary_float",
    "Atan": "unary_float",
    "Asinh": "unary_float",
    "Acosh": "unary_float",
    "Atanh": "unary_float",
    # Activations
    "Relu": "unary_float",
    "Sigmoid": "unary_float",
    "Softplus": "unary_float",
    "Softsign": "unary_float",
    "Mish": "unary_float",
    "Gelu": "gelu",
    "Celu": "celu",
    "Elu": "elu",
    "Selu": "unary_float",
    "LeakyRelu": "leaky_relu",
    "HardSigmoid": "unary_float",
    "HardSwish": "unary_float",
    "Hardmax": "hardmax",
    "Softmax": "softmax",
    "LogSoftmax": "log_softmax",
    "PRelu": "prelu",
    "ThresholdedRelu": "thresholded_relu",
    "Swish": "unary_float",
    # Comparison
    "Equal": "comparison",
    "Greater": "comparison",
    "Less": "comparison",
    "GreaterOrEqual": "comparison",
    "LessOrEqual": "comparison",
    # Logical
    "And": "logical_binary",
    "Or": "logical_binary",
    "Xor": "logical_binary",
    "Not": "logical_unary",
    # Bitwise
    "BitwiseAnd": "bitwise_binary",
    "BitwiseOr": "bitwise_binary",
    "BitwiseXor": "bitwise_binary",
    "BitwiseNot": "bitwise_unary",
    "BitShift": "bitshift",
    # Reduction
    "ReduceMax": "reduce",
    "ReduceMin": "reduce",
    "ReduceMean": "reduce",
    "ReduceSum": "reduce",
    "ReduceProd": "reduce",
    "ReduceL1": "reduce",
    "ReduceL2": "reduce",
    "ReduceLogSum": "reduce",
    "ReduceLogSumExp": "reduce",
    "ReduceSumSquare": "reduce",
    "ArgMax": "argmax",
    "ArgMin": "argmin",
    # Shape/manipulation
    "Reshape": "reshape",
    "Transpose": "transpose",
    "Flatten": "flatten",
    "Squeeze": "squeeze",
    "Unsqueeze": "unsqueeze",
    "Concat": "concat",
    "Split": "split",
    "Gather": "gather",
    "GatherElements": "gather_elements",
    "GatherND": "gather_nd",
    "Slice": "slice_op",
    "Tile": "tile",
    "Expand": "expand",
    "Pad": "pad",
    "Clip": "clip",
    "Shape": "shape_op",
    "Size": "size_op",
    "DepthToSpace": "depth_to_space",
    "SpaceToDepth": "space_to_depth",
    "ScatterElements": "scatter_elements",
    "ScatterND": "scatter_nd",
    # Matrix
    "MatMul": "matmul",
    "Gemm": "gemm",
    "MatMulInteger": "matmul_integer",
    # Conv (generates Conv2d test)
    "Conv": "conv",
    "ConvTranspose": "conv_transpose",
    # Pooling
    "AveragePool": "avg_pool",
    "MaxPool": "max_pool",
    "GlobalAveragePool": "global_avg_pool",
    # Normalization
    "BatchNormalization": "batch_norm",
    "InstanceNormalization": "instance_norm",
    "LayerNormalization": "layer_norm",
    "GroupNormalization": "group_norm",
    # Utility
    "Dropout": "dropout",
    "Identity": "identity",
    "Cast": "cast",
    "Where": "where_op_gen",
    "NonZero": "nonzero",
    "Constant": "constant",
    "ConstantOfShape": "constant_of_shape",
    "OneHot": "onehot",
    "TopK": "topk",
    "IsInf": "isinf",
    "IsNaN": "isnan",
    "Trilu": "trilu",
    "CumSum": "cumsum",
    "EyeLike": "eyelike",
    "Range": "range_op",
    "Resize": "resize",
    "GridSample": "grid_sample",
    # Random
    "RandomNormal": "random_normal",
    "RandomUniform": "random_uniform",
    "RandomNormalLike": "random_like",
    "RandomUniformLike": "random_like",
    "Bernoulli": "bernoulli",
    # RNN
    "LSTM": "lstm",
    "GRU": "gru",
    # Control flow
    "If": "if_op",
    # DeformConv
    "DeformConv": "deform_conv",
}


# ---------------------------------------------------------------------------
# Parse specs to find version history
# ---------------------------------------------------------------------------


def parse_spec_versions(spec_dir: Path) -> dict[str, list[int]]:
    """Parse onnx-spec/ops/*.md to extract version lists."""
    versions = {}
    for md_file in sorted(spec_dir.glob("*.md")):
        op_name = md_file.stem
        text = md_file.read_text()
        m = re.search(r"All versions:\s*([\d, ]+)", text)
        if m:
            ver_list = [int(v.strip()) for v in m.group(1).split(",")]
            versions[op_name] = sorted(ver_list)
        else:
            # Some ops only have "First introduced in opset **N**" with no updates
            m2 = re.search(r"First introduced in opset \*\*(\d+)\*\*", text)
            if m2:
                versions[op_name] = [int(m2.group(1))]
    return versions


# ---------------------------------------------------------------------------
# Op generators - each returns (nodes, inputs, outputs, initializers)
# ---------------------------------------------------------------------------

# Prefix helper: ensures unique names per op in a multi-op graph
def _p(op_name: str, name: str) -> str:
    return f"{op_name.lower()}_{name}"


def make_unary_float(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_binary_float(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_variadic_float(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_comparison(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.BOOL, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_logical_binary(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.BOOL, [2, 3, 4])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.BOOL, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.BOOL, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_logical_unary(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.BOOL, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.BOOL, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_bitwise_binary(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.INT32, [2, 3, 4])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.INT32, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT32, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_bitwise_unary(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.INT32, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT32, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_bitshift(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.UINT32, [2, 3])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.UINT32, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.UINT32, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"), direction="LEFT")
    return [node], [a, b], [out], []


def make_reduce(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 18:
        # Axes as input, keepdims as attribute
        axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=_p(op_name, "axes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "axes")], [_p(op_name, "output")], name=_p(op_name, "node"), keepdims=1)
        return [node], [inp], [out], [axes_init]
    elif opset >= 13:
        # Axes as input (opset 13 for ReduceSum, but other reduces still use attrs until 18)
        # For ReduceSum opset 13+, axes is an input. For others, axes is still an attribute.
        if op_name == "ReduceSum":
            axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=_p(op_name, "axes"))
            node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "axes")], [_p(op_name, "output")], name=_p(op_name, "node"), keepdims=1)
            return [node], [inp], [out], [axes_init]
        else:
            node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axes=[1], keepdims=1)
            return [node], [inp], [out], []
    else:
        # All axes as attribute
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axes=[1], keepdims=1)
        return [node], [inp], [out], []


def make_argmax(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT64, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=1, keepdims=1)
    return [node], [inp], [out], []


def make_argmin(op_name: str, opset: int):
    return make_argmax(op_name, opset)


def make_reshape(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 5:
        shape_init = numpy_helper.from_array(np.array([6, 4], dtype=np.int64), name=_p(op_name, "shape"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "shape")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [shape_init]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), shape=[6, 4])
        return [node], [inp], [out], []


def make_transpose(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [4, 3, 2])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), perm=[2, 1, 0])
    return [node], [inp], [out], []


def make_flatten(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=1)
    return [node], [inp], [out], []


def make_squeeze(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 1, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 13:
        axes_init = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name=_p(op_name, "axes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "axes")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [axes_init]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axes=[0, 2])
        return [node], [inp], [out], []


def make_unsqueeze(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 13:
        axes_init = numpy_helper.from_array(np.array([0, 3], dtype=np.int64), name=_p(op_name, "axes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "axes")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [axes_init]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), axes=[0, 3])
        return [node], [inp], [out], []


def make_concat(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [4, 3])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=0)
    return [node], [a, b], [out], []


def make_split(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 6])
    out1 = helper.make_tensor_value_info(_p(op_name, "output_0"), TensorProto.FLOAT, None)
    out2 = helper.make_tensor_value_info(_p(op_name, "output_1"), TensorProto.FLOAT, None)
    if opset >= 13:
        split_init = numpy_helper.from_array(np.array([3, 3], dtype=np.int64), name=_p(op_name, "split_sizes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "split_sizes")], [_p(op_name, "output_0"), _p(op_name, "output_1")], name=_p(op_name, "node"), axis=1)
        return [node], [inp], [out1, out2], [split_init]
    elif opset >= 2:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output_0"), _p(op_name, "output_1")], name=_p(op_name, "node"), axis=1, split=[3, 3])
        return [node], [inp], [out1, out2], []
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output_0"), _p(op_name, "output_1")], name=_p(op_name, "node"), axis=1, split=[3, 3])
        return [node], [inp], [out1, out2], []


def make_gather(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 4])
    indices_init = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name=_p(op_name, "indices"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "indices")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=0)
    return [node], [inp], [out], [indices_init]


def make_gather_elements(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    indices_init = numpy_helper.from_array(np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64), name=_p(op_name, "indices"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "indices")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=0)
    return [node], [inp], [out], [indices_init]


def make_gather_nd(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    indices_init = numpy_helper.from_array(np.array([[0, 1], [1, 0]], dtype=np.int64), name=_p(op_name, "indices"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "indices")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], [indices_init]


def make_slice_op(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [4, 6])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 10:
        starts = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name=_p(op_name, "starts"))
        ends = numpy_helper.from_array(np.array([2, 4], dtype=np.int64), name=_p(op_name, "ends"))
        axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name=_p(op_name, "axes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "starts"), _p(op_name, "ends"), _p(op_name, "axes")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [starts, ends, axes]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), starts=[0, 1], ends=[2, 4], axes=[0, 1])
        return [node], [inp], [out], []


def make_tile(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    repeats = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=_p(op_name, "repeats"))
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "repeats")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], [repeats]


def make_expand(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    shape = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=_p(op_name, "shape"))
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "shape")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], [shape]


def make_pad(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 11:
        pads = numpy_helper.from_array(np.array([0, 1, 0, 1], dtype=np.int64), name=_p(op_name, "pads"))
        const_val = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=_p(op_name, "constant_value"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "pads"), _p(op_name, "constant_value")], [_p(op_name, "output")], name=_p(op_name, "node"), mode="constant")
        return [node], [inp], [out], [pads, const_val]
    elif opset >= 2:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), mode="constant", pads=[0, 1, 0, 1], value=0.0)
        return [node], [inp], [out], []
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), mode="constant", paddings=[0, 1, 0, 1], value=0.0)
        return [node], [inp], [out], []


def make_clip(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    if opset >= 11:
        min_val = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=_p(op_name, "min"))
        max_val = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name=_p(op_name, "max"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "min"), _p(op_name, "max")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [min_val, max_val]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), **{"min": 0.0, "max": 6.0})
        return [node], [inp], [out], []


def make_shape_op(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT64, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_size_op(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT64, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_depth_to_space(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 8, 2, 2])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), blocksize=2, mode="DCR")
    return [node], [inp], [out], []


def make_space_to_depth(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 2, 4, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), blocksize=2)
    return [node], [inp], [out], []


def make_scatter_elements(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 3])
    indices_init = numpy_helper.from_array(np.array([[0, 1, 2]], dtype=np.int64), name=_p(op_name, "indices"))
    updates = helper.make_tensor_value_info(_p(op_name, "updates"), TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "indices"), _p(op_name, "updates")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=0)
    return [node], [inp, updates], [out], [indices_init]


def make_scatter_nd(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [4, 4])
    indices_init = numpy_helper.from_array(np.array([[0], [2]], dtype=np.int64), name=_p(op_name, "indices"))
    updates = helper.make_tensor_value_info(_p(op_name, "updates"), TensorProto.FLOAT, [2, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "indices"), _p(op_name, "updates")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp, updates], [out], [indices_init]


def make_matmul(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.FLOAT, [3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_gemm(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.FLOAT, [2, 3])
    b_init = numpy_helper.from_array(np.ones([3, 4], dtype=np.float32), name=_p(op_name, "b"))
    c_init = numpy_helper.from_array(np.zeros([4], dtype=np.float32), name=_p(op_name, "c"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b"), _p(op_name, "c")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a], [out], [b_init, c_init]


def make_matmul_integer(op_name: str, opset: int):
    a = helper.make_tensor_value_info(_p(op_name, "a"), TensorProto.UINT8, [2, 3])
    b = helper.make_tensor_value_info(_p(op_name, "b"), TensorProto.UINT8, [3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT32, [2, 4])
    node = helper.make_node(op_name, [_p(op_name, "a"), _p(op_name, "b")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [a, b], [out], []


def make_conv(op_name: str, opset: int):
    """Generate a 2D convolution node."""
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 5, 5])
    w_init = numpy_helper.from_array(np.ones([2, 3, 3, 3], dtype=np.float32), name=_p(op_name, "weight"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "weight")], [_p(op_name, "output")], name=_p(op_name, "node"), kernel_shape=[3, 3])
    return [node], [inp], [out], [w_init]


def make_conv_transpose(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 5, 5])
    w_init = numpy_helper.from_array(np.ones([3, 2, 3, 3], dtype=np.float32), name=_p(op_name, "weight"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "weight")], [_p(op_name, "output")], name=_p(op_name, "node"), kernel_shape=[3, 3])
    return [node], [inp], [out], [w_init]


def make_avg_pool(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), kernel_shape=[2, 2], strides=[2, 2])
    return [node], [inp], [out], []


def make_max_pool(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), kernel_shape=[2, 2], strides=[2, 2])
    return [node], [inp], [out], []


def make_global_avg_pool(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_batch_norm(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 4, 4])
    scale = numpy_helper.from_array(np.ones([3], dtype=np.float32), name=_p(op_name, "scale"))
    bias = numpy_helper.from_array(np.zeros([3], dtype=np.float32), name=_p(op_name, "bias"))
    mean = numpy_helper.from_array(np.zeros([3], dtype=np.float32), name=_p(op_name, "mean"))
    var = numpy_helper.from_array(np.ones([3], dtype=np.float32), name=_p(op_name, "var"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    inputs = [_p(op_name, "input"), _p(op_name, "scale"), _p(op_name, "bias"), _p(op_name, "mean"), _p(op_name, "var")]
    outputs = [_p(op_name, "output")]
    extra_outputs = []
    kwargs = {}
    if opset < 9:
        # Opset <9 required 5 outputs; add placeholder names for unused ones
        for suffix in ["mean_out", "var_out", "saved_mean", "saved_var"]:
            name = _p(op_name, suffix)
            outputs.append(name)
            extra_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    if opset < 6:
        kwargs["spatial"] = 1
    elif opset < 9:
        kwargs["consumed_inputs"] = [0, 0, 0, 1, 1]
    node = helper.make_node(op_name, inputs, outputs, name=_p(op_name, "node"), **kwargs)
    return [node], [inp], [out] + extra_outputs, [scale, bias, mean, var]


def make_instance_norm(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 3, 4, 4])
    scale = numpy_helper.from_array(np.ones([3], dtype=np.float32), name=_p(op_name, "scale"))
    bias = numpy_helper.from_array(np.zeros([3], dtype=np.float32), name=_p(op_name, "bias"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(
        op_name, [_p(op_name, "input"), _p(op_name, "scale"), _p(op_name, "bias")], [_p(op_name, "output")], name=_p(op_name, "node")
    )
    return [node], [inp], [out], [scale, bias]


def make_layer_norm(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    scale = numpy_helper.from_array(np.ones([4], dtype=np.float32), name=_p(op_name, "scale"))
    bias = numpy_helper.from_array(np.zeros([4], dtype=np.float32), name=_p(op_name, "bias"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(
        op_name, [_p(op_name, "input"), _p(op_name, "scale"), _p(op_name, "bias")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=-1
    )
    return [node], [inp], [out], [scale, bias]


def make_group_norm(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 4, 3, 3])
    scale = numpy_helper.from_array(np.ones([4], dtype=np.float32), name=_p(op_name, "scale"))
    bias = numpy_helper.from_array(np.zeros([4], dtype=np.float32), name=_p(op_name, "bias"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(
        op_name, [_p(op_name, "input"), _p(op_name, "scale"), _p(op_name, "bias")], [_p(op_name, "output")], name=_p(op_name, "node"), num_groups=2
    )
    return [node], [inp], [out], [scale, bias]


def make_dropout(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    if opset >= 12:
        ratio = numpy_helper.from_array(np.array(0.5, dtype=np.float32), name=_p(op_name, "ratio"))
        training = numpy_helper.from_array(np.array(False, dtype=bool), name=_p(op_name, "training_mode"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "ratio"), _p(op_name, "training_mode")], [_p(op_name, "output")], name=_p(op_name, "node"))
        return [node], [inp], [out], [ratio, training]
    elif opset >= 7:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), ratio=0.5)
        return [node], [inp], [out], []
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), ratio=0.5)
        return [node], [inp], [out], []


def make_identity(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_cast(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT32, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), to=TensorProto.INT32)
    return [node], [inp], [out], []


def make_where_op_gen(op_name: str, opset: int):
    cond = helper.make_tensor_value_info(_p(op_name, "condition"), TensorProto.BOOL, [2, 3])
    x = helper.make_tensor_value_info(_p(op_name, "x"), TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info(_p(op_name, "y"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "condition"), _p(op_name, "x"), _p(op_name, "y")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [cond, x, y], [out], []


def make_nonzero(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.INT64, None)
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_constant(op_name: str, opset: int):
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    val = numpy_helper.from_array(np.ones([2, 3], dtype=np.float32))
    node = helper.make_node(op_name, [], [_p(op_name, "output")], name=_p(op_name, "node"), value=val)
    return [node], [], [out], []


def make_constant_of_shape(op_name: str, opset: int):
    shape_init = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=_p(op_name, "shape"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    val = numpy_helper.from_array(np.array([1.0], dtype=np.float32))
    node = helper.make_node(op_name, [_p(op_name, "shape")], [_p(op_name, "output")], name=_p(op_name, "node"), value=val)
    return [node], [], [out], [shape_init]


def make_onehot(op_name: str, opset: int):
    indices = helper.make_tensor_value_info(_p(op_name, "indices"), TensorProto.INT64, [3])
    depth_init = numpy_helper.from_array(np.array(5, dtype=np.int64), name=_p(op_name, "depth"))
    values_init = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name=_p(op_name, "values"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "indices"), _p(op_name, "depth"), _p(op_name, "values")], [_p(op_name, "output")], name=_p(op_name, "node"), axis=-1)
    return [node], [indices], [out], [depth_init, values_init]


def make_topk(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 4])
    out_values = helper.make_tensor_value_info(_p(op_name, "values"), TensorProto.FLOAT, None)
    out_indices = helper.make_tensor_value_info(_p(op_name, "indices"), TensorProto.INT64, None)
    if opset >= 10:
        k_init = numpy_helper.from_array(np.array([2], dtype=np.int64), name=_p(op_name, "k"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "k")], [_p(op_name, "values"), _p(op_name, "indices")], name=_p(op_name, "node"))
        return [node], [inp], [out_values, out_indices], [k_init]
    else:
        node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "values"), _p(op_name, "indices")], name=_p(op_name, "node"), k=2)
        return [node], [inp], [out_values, out_indices], []


def make_isinf(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.BOOL, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_isnan(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.BOOL, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_trilu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), upper=1)
    return [node], [inp], [out], []


def make_cumsum(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    axis_init = numpy_helper.from_array(np.array(1, dtype=np.int32), name=_p(op_name, "axis"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "axis")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], [axis_init]


def make_eyelike(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [3, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [3, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_range_op(op_name: str, opset: int):
    start = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=_p(op_name, "start"))
    limit = numpy_helper.from_array(np.array(10.0, dtype=np.float32), name=_p(op_name, "limit"))
    delta = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name=_p(op_name, "delta"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "start"), _p(op_name, "limit"), _p(op_name, "delta")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [], [out], [start, limit, delta]


def make_resize(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 1, 2, 2])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    if opset >= 11:
        roi = numpy_helper.from_array(np.array([], dtype=np.float32), name=_p(op_name, "roi"))
        scales = numpy_helper.from_array(np.array([], dtype=np.float32), name=_p(op_name, "scales"))
        sizes = numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.int64), name=_p(op_name, "sizes"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "roi"), _p(op_name, "scales"), _p(op_name, "sizes")], [_p(op_name, "output")], name=_p(op_name, "node"), mode="nearest")
        return [node], [inp], [out], [roi, scales, sizes]
    else:
        scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name=_p(op_name, "scales"))
        node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "scales")], [_p(op_name, "output")], name=_p(op_name, "node"), mode="nearest")
        return [node], [inp], [out], [scales]


def make_grid_sample(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 1, 3, 3])
    grid = helper.make_tensor_value_info(_p(op_name, "grid"), TensorProto.FLOAT, [1, 2, 2, 2])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "grid")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp, grid], [out], []


def make_random_normal(op_name: str, opset: int):
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_name, [], [_p(op_name, "output")], name=_p(op_name, "node"), shape=[2, 3], mean=0.0, scale=1.0)
    return [node], [], [out], []


def make_random_uniform(op_name: str, opset: int):
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_name, [], [_p(op_name, "output")], name=_p(op_name, "node"), shape=[2, 3], low=0.0, high=1.0)
    return [node], [], [out], []


def make_random_like(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_bernoulli(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], []


def make_lstm(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 2, 3])
    w_init = numpy_helper.from_array(np.zeros([1, 16, 3], dtype=np.float32), name=_p(op_name, "W"))
    r_init = numpy_helper.from_array(np.zeros([1, 16, 4], dtype=np.float32), name=_p(op_name, "R"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "W"), _p(op_name, "R")], [_p(op_name, "output")], name=_p(op_name, "node"), hidden_size=4)
    return [node], [inp], [out], [w_init, r_init]


def make_gru(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 2, 3])
    w_init = numpy_helper.from_array(np.zeros([1, 12, 3], dtype=np.float32), name=_p(op_name, "W"))
    r_init = numpy_helper.from_array(np.zeros([1, 12, 4], dtype=np.float32), name=_p(op_name, "R"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "W"), _p(op_name, "R")], [_p(op_name, "output")], name=_p(op_name, "node"), hidden_size=4)
    return [node], [inp], [out], [w_init, r_init]


def make_if_op(op_name: str, opset: int):
    cond = helper.make_tensor_value_info(_p(op_name, "condition"), TensorProto.BOOL, [])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3])

    then_out = helper.make_tensor_value_info(_p(op_name, "then_out"), TensorProto.FLOAT, [2, 3])
    then_const = numpy_helper.from_array(np.ones([2, 3], dtype=np.float32))
    then_node = helper.make_node("Constant", [], [_p(op_name, "then_out")], value=then_const)
    then_graph = helper.make_graph([then_node], "then_graph", [], [then_out])

    else_out = helper.make_tensor_value_info(_p(op_name, "else_out"), TensorProto.FLOAT, [2, 3])
    else_const = numpy_helper.from_array(np.zeros([2, 3], dtype=np.float32))
    else_node = helper.make_node("Constant", [], [_p(op_name, "else_out")], value=else_const)
    else_graph = helper.make_graph([else_node], "else_graph", [], [else_out])

    node = helper.make_node(op_name, [_p(op_name, "condition")], [_p(op_name, "output")], name=_p(op_name, "node"), then_branch=then_graph, else_branch=else_graph)
    return [node], [cond], [out], []


def make_deform_conv(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [1, 1, 3, 3])
    w_init = numpy_helper.from_array(np.ones([1, 1, 2, 2], dtype=np.float32), name=_p(op_name, "weight"))
    offset = helper.make_tensor_value_info(_p(op_name, "offset"), TensorProto.FLOAT, [1, 8, 2, 2])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, None)
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "weight"), _p(op_name, "offset")], [_p(op_name, "output")], name=_p(op_name, "node"), kernel_shape=[2, 2])
    return [node], [inp, offset], [out], [w_init]


def make_softmax(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    kwargs = {}
    if opset >= 13:
        kwargs["axis"] = -1
    else:
        kwargs["axis"] = 1
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), **kwargs)
    return [node], [inp], [out], []


def make_log_softmax(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    kwargs = {}
    if opset >= 13:
        kwargs["axis"] = -1
    else:
        kwargs["axis"] = 1
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), **kwargs)
    return [node], [inp], [out], []


def make_hardmax(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    kwargs = {}
    if opset >= 13:
        kwargs["axis"] = -1
    else:
        kwargs["axis"] = 1
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), **kwargs)
    return [node], [inp], [out], []


def make_gelu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), approximate="none")
    return [node], [inp], [out], []


def make_celu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), alpha=1.0)
    return [node], [inp], [out], []


def make_elu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), alpha=1.0)
    return [node], [inp], [out], []


def make_leaky_relu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), alpha=0.01)
    return [node], [inp], [out], []


def make_prelu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    slope_init = numpy_helper.from_array(np.array([0.01], dtype=np.float32), name=_p(op_name, "slope"))
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input"), _p(op_name, "slope")], [_p(op_name, "output")], name=_p(op_name, "node"))
    return [node], [inp], [out], [slope_init]


def make_thresholded_relu(op_name: str, opset: int):
    inp = helper.make_tensor_value_info(_p(op_name, "input"), TensorProto.FLOAT, [2, 3, 4])
    out = helper.make_tensor_value_info(_p(op_name, "output"), TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_name, [_p(op_name, "input")], [_p(op_name, "output")], name=_p(op_name, "node"), alpha=1.0)
    return [node], [inp], [out], []


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

GENERATORS = {
    "unary_float": make_unary_float,
    "binary_float": make_binary_float,
    "variadic_float": make_variadic_float,
    "comparison": make_comparison,
    "logical_binary": make_logical_binary,
    "logical_unary": make_logical_unary,
    "bitwise_binary": make_bitwise_binary,
    "bitwise_unary": make_bitwise_unary,
    "bitshift": make_bitshift,
    "reduce": make_reduce,
    "argmax": make_argmax,
    "argmin": make_argmin,
    "reshape": make_reshape,
    "transpose": make_transpose,
    "flatten": make_flatten,
    "squeeze": make_squeeze,
    "unsqueeze": make_unsqueeze,
    "concat": make_concat,
    "split": make_split,
    "gather": make_gather,
    "gather_elements": make_gather_elements,
    "gather_nd": make_gather_nd,
    "slice_op": make_slice_op,
    "tile": make_tile,
    "expand": make_expand,
    "pad": make_pad,
    "clip": make_clip,
    "shape_op": make_shape_op,
    "size_op": make_size_op,
    "depth_to_space": make_depth_to_space,
    "space_to_depth": make_space_to_depth,
    "scatter_elements": make_scatter_elements,
    "scatter_nd": make_scatter_nd,
    "matmul": make_matmul,
    "gemm": make_gemm,
    "matmul_integer": make_matmul_integer,
    "conv": make_conv,
    "conv_transpose": make_conv_transpose,
    "avg_pool": make_avg_pool,
    "max_pool": make_max_pool,
    "global_avg_pool": make_global_avg_pool,
    "batch_norm": make_batch_norm,
    "instance_norm": make_instance_norm,
    "layer_norm": make_layer_norm,
    "group_norm": make_group_norm,
    "dropout": make_dropout,
    "identity": make_identity,
    "cast": make_cast,
    "where_op_gen": make_where_op_gen,
    "nonzero": make_nonzero,
    "constant": make_constant,
    "constant_of_shape": make_constant_of_shape,
    "onehot": make_onehot,
    "topk": make_topk,
    "isinf": make_isinf,
    "isnan": make_isnan,
    "trilu": make_trilu,
    "cumsum": make_cumsum,
    "eyelike": make_eyelike,
    "range_op": make_range_op,
    "resize": make_resize,
    "grid_sample": make_grid_sample,
    "random_normal": make_random_normal,
    "random_uniform": make_random_uniform,
    "random_like": make_random_like,
    "bernoulli": make_bernoulli,
    "lstm": make_lstm,
    "gru": make_gru,
    "if_op": make_if_op,
    "deform_conv": make_deform_conv,
    "softmax": make_softmax,
    "log_softmax": make_log_softmax,
    "hardmax": make_hardmax,
    "gelu": make_gelu,
    "celu": make_celu,
    "elu": make_elu,
    "leaky_relu": make_leaky_relu,
    "prelu": make_prelu,
    "thresholded_relu": make_thresholded_relu,
}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def build_opset_map(spec_versions: dict[str, list[int]]) -> dict[int, list[str]]:
    """Build mapping: opset_version -> list of supported ops changed at that version."""
    opset_map: dict[int, list[str]] = {}
    for op_name, gen_key in SUPPORTED_OPS.items():
        if op_name not in spec_versions:
            print(f"  WARNING: {op_name} not found in specs, skipping", file=sys.stderr)
            continue
        versions = spec_versions[op_name]
        for v in versions:
            opset_map.setdefault(v, []).append(op_name)
    return opset_map


def generate_model(opset: int, ops: list[str]) -> Optional[onnx.ModelProto]:
    """Generate one ONNX model containing all given ops at the specified opset."""
    all_nodes = []
    all_inputs = []
    all_outputs = []
    all_initializers = []
    generated = []

    for op_name in sorted(ops):
        gen_key = SUPPORTED_OPS[op_name]
        gen_func = GENERATORS.get(gen_key)
        if gen_func is None:
            print(f"  WARNING: No generator for {op_name} ({gen_key}), skipping", file=sys.stderr)
            continue
        try:
            nodes, inputs, outputs, initializers = gen_func(op_name, opset)
            all_nodes.extend(nodes)
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            all_initializers.extend(initializers)
            generated.append(op_name)
        except Exception as e:
            print(f"  WARNING: Failed to generate {op_name} at opset {opset}: {e}", file=sys.stderr)

    if not all_nodes:
        return None

    graph = helper.make_graph(
        all_nodes,
        f"opset_{opset}_compliance",
        all_inputs,
        all_outputs,
        initializer=all_initializers,
    )

    # Use appropriate IR version for the opset
    # ir_version must be compatible with opset version
    if opset >= 21:
        ir_version = 10
    elif opset >= 19:
        ir_version = 9
    elif opset >= 16:
        ir_version = 8
    elif opset >= 12:
        ir_version = 7
    elif opset >= 10:
        ir_version = 6
    elif opset >= 8:
        ir_version = 4
    else:
        ir_version = 3

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=ir_version,
    )

    return model, generated


def main():
    print("Parsing ONNX specs...")
    spec_versions = parse_spec_versions(SPEC_DIR)
    print(f"  Found {len(spec_versions)} operator specs")

    print("Building opset map...")
    opset_map = build_opset_map(spec_versions)

    # Sort by opset version
    sorted_opsets = sorted(opset_map.keys())
    print(f"  Opset versions with supported op changes: {sorted_opsets}")

    # Clean stale fixtures from previous runs
    if FIXTURE_DIR.exists():
        for f in FIXTURE_DIR.glob("*.onnx"):
            f.unlink()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Import MIN_OPSET from the Rust test generator for splitting
    from gen_rust_tests import MIN_OPSET

    # Generate a manifest file for Rust tests to read
    manifest_lines = []

    total_pass = 0
    total_fail = 0
    for opset in sorted_opsets:
        ops = opset_map[opset]

        # Split ops into passing (min_opset <= opset) and failing (min_opset > opset)
        passing = [o for o in ops if MIN_OPSET.get(o, 999) <= opset]
        failing = [o for o in ops if MIN_OPSET.get(o, 999) > opset]

        print(f"\nOpset {opset:2d}: {len(passing)} passing, {len(failing)} failing")

        # Generate passing model
        if passing:
            result = generate_model(opset, passing)
            if result is not None:
                model, generated = result
                filename = f"opset_{opset:02d}.onnx"
                onnx.save(model, str(FIXTURE_DIR / filename))
                total_pass += len(generated)
                print(f"  {filename}: {', '.join(generated)}")

        # Generate failing model (for ops where min_opset > opset)
        if failing:
            result = generate_model(opset, failing)
            if result is not None:
                model, generated = result
                filename = f"opset_{opset:02d}_unsupported.onnx"
                onnx.save(model, str(FIXTURE_DIR / filename))
                total_fail += len(generated)
                print(f"  {filename}: {', '.join(generated)}")

        # Manifest line: opset:passing_ops|failing_ops
        manifest_lines.append(
            f"{opset}:{','.join(sorted(passing))}|{','.join(sorted(failing))}"
        )

    # Write manifest
    manifest_path = FIXTURE_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nManifest written to {manifest_path}")
    print(f"Total: {total_pass} passing + {total_fail} failing = {total_pass + total_fail} op-version combinations")


if __name__ == "__main__":
    main()
