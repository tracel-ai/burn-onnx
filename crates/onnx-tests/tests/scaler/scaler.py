#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate models: scaler.onnx, scaler_per_feature_3d.onnx, scaler_i64.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 1


def make_model(node, input_info, output_info):
    graph = helper.make_graph(
        [node],
        "scaler_test",
        [input_info],
        [output_info],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),
        ],
    )


def gen_scaler():
    """2D F32, scalar scale/offset — baseline test."""
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    node = helper.make_node(
        "Scaler",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        scale=[2.0],
        offset=[1.0],
    )
    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3]),
    )
    onnx.save(model, "scaler.onnx")

    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})[0]
    print("scaler.onnx output:\n", result)


def gen_scaler_per_feature_3d():
    """Rank-3 F32, per-feature scale/offset — exercises the [1,...,1,F] reshape broadcast."""
    # shape [2, 2, 3]: 2 batches, 2 timesteps, 3 features
    input_data = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
        dtype=np.float32,
    )
    scale = [1.0, 2.0, 0.5]
    offset = [0.0, 1.0, 2.0]

    node = helper.make_node(
        "Scaler",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        scale=scale,
        offset=offset,
    )
    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2, 3]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2, 3]),
    )
    onnx.save(model, "scaler_per_feature_3d.onnx")

    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})[0]
    print("scaler_per_feature_3d.onnx output:\n", result)
    expected = (input_data - np.array(offset)) * np.array(scale)
    print("expected:\n", expected)


def gen_scaler_i64():
    """2D I64, per-feature scale/offset — exercises .int().cast(DType::F32) codegen path.
    Values chosen to produce integer results, avoiding ReferenceEvaluator truncation issues."""
    # shape [2, 3]
    input_data = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.int64)
    # scale and offset chosen so (x - offset) * scale is always an integer
    scale = [3.0, 2.0, 1.0]
    offset = [2.0, 2.0, 2.0]
    # Expected: [(2-2)*3, (4-2)*2, (6-2)*1] = [0, 4, 4]
    #           [(8-2)*3, (10-2)*2, (12-2)*1] = [18, 16, 10]

    node = helper.make_node(
        "Scaler",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        scale=scale,
        offset=offset,
    )
    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.INT64, [2, 3]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3]),
    )
    onnx.save(model, "scaler_i64.onnx")

    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})[0]
    print("scaler_i64.onnx output:\n", result)
    expected = (input_data.astype(np.float32) - np.array(offset, dtype=np.float32)) * np.array(scale, dtype=np.float32)
    print("expected:\n", expected)


def main():
    gen_scaler()
    gen_scaler_per_feature_3d()
    gen_scaler_i64()


if __name__ == "__main__":
    main()
