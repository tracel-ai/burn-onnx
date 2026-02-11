#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/pad/pad_runtime_constant.onnx
# Tests Pad with constant_value as a runtime graph input (not a static initializer).

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.checker import check_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)


def main() -> None:
    # Pad with constant_value as a runtime input (not an initializer).
    # This tests the case from issue #101 where constant_value comes from
    # another node's output rather than being a static constant.

    # Graph inputs: data tensor and constant_value scalar
    inputs = [
        make_tensor_value_info("input_tensor", TensorProto.FLOAT, [None, None]),
        make_tensor_value_info("constant_value", TensorProto.FLOAT, []),
    ]
    outputs = [make_tensor_value_info("output", TensorProto.FLOAT, [None, None])]

    # Pads as an initializer (static), but constant_value is a graph input (runtime)
    pads = numpy_helper.from_array(np.array([1, 1, 1, 1]).astype(np.int64), name="pads")

    node = make_node(
        "Pad",
        inputs=["input_tensor", "pads", "constant_value"],
        outputs=["output"],
        mode="constant",
    )

    graph = make_graph(
        nodes=[node],
        name="PadRuntimeConstantGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=[pads],
    )

    onnx_model = make_model(graph)
    check_model(onnx_model)

    test_runtime_constant(onnx_model)

    onnx.save(onnx_model, Path(__file__).with_name("pad_runtime_constant.onnx"))
    print("Generated pad_runtime_constant.onnx")


def test_runtime_constant(model) -> None:
    sess = ReferenceEvaluator(model)

    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    constant_value = np.float32(5.0)

    result = sess.run(
        None, {"input_tensor": input_tensor, "constant_value": constant_value}
    )[0]

    expected = np.array(
        [
            [5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 2.0, 5.0],
            [5.0, 3.0, 4.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )

    if not np.allclose(result, expected):
        print(f"Expected:\n{expected}")
        print(f"Got:\n{result}")
        raise Exception("Runtime constant padding test failed")

    print("Runtime constant padding test passed!")


if __name__ == "__main__":
    main()
