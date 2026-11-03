#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///

"""Keep a runtime shape alive across partitions before allocating a float tensor."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    np.random.seed(42)
    input_name = "execution_device"
    nodes = [helper.make_node("Shape", [input_name], ["shape0"])]
    # Negations remain shape operations; later partitions have no tensor input
    # to determine an allocation device. An even count preserves shape.
    for i in range(250):
        nodes.append(helper.make_node("Neg", [f"shape{i}"], [f"shape{i + 1}"]))
    nodes.extend(
        [
            helper.make_node("Cast", ["shape250"], ["floats"], to=TensorProto.FLOAT),
            helper.make_node("Sqrt", ["floats"], ["roots"]),
            helper.make_node("Mul", ["roots", "scale"], ["output"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "partitioned_device",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["b", "d"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        initializer=[
            numpy_helper.from_array(np.array([2.0, 3.0], dtype=np.float32), "scale")
        ],
    )
    model = helper.make_model(
        graph, ir_version=8, opset_imports=[helper.make_opsetid("", 18)]
    )
    onnx.checker.check_model(model)
    input_data = np.random.randn(2, 4).astype(np.float32)
    output = ReferenceEvaluator(model).run(None, {input_name: input_data})[0]
    np.testing.assert_allclose(output, np.sqrt([2.0, 4.0]) * [2.0, 3.0], rtol=1e-6)
    onnx.save(model, Path(__file__).with_suffix(".onnx"))
    print(output)


if __name__ == "__main__":
    main()
