#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/slice/slice_tensor_to_split.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [3, 6],
    )

    output_0 = helper.make_tensor_value_info(
        "output_0",
        TensorProto.FLOAT,
        [3, 2],
    )
    output_1 = helper.make_tensor_value_info(
        "output_1",
        TensorProto.FLOAT,
        [3, 2],
    )

    starts = helper.make_tensor(
        name="starts",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2],
    )
    ends = helper.make_tensor(
        name="ends",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[np.iinfo(np.int64).max],
    )
    axes = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],
    )
    steps = helper.make_tensor(
        name="steps",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],
    )
    split_sizes = helper.make_tensor(
        name="split_sizes",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[2, 2],
    )

    slice_node = helper.make_node(
        "Slice",
        inputs=["input", "starts", "ends", "axes", "steps"],
        outputs=["sliced"],
        name="slice",
    )

    split_node = helper.make_node(
        "Split",
        inputs=["sliced", "split_sizes"],
        outputs=["output_0", "output_1"],
        name="split",
        axis=1,
    )

    graph = helper.make_graph(
        nodes=[slice_node, split_node],
        name="slice_tensor_to_split",
        inputs=[input_tensor],
        outputs=[output_0, output_1],
        initializer=[starts, ends, axes, steps, split_sizes],
    )

    model = helper.make_model(
        graph,
        producer_name="slice_tensor_to_split_generator",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    onnx.checker.check_model(model)
    onnx.save(model, "slice_tensor_to_split.onnx")

    session = ReferenceEvaluator(model)

    input_data = np.arange(18, dtype=np.float32).reshape(3, 6)
    outputs = session.run(None, {"input": input_data})

    print("Finished exporting model to slice_tensor_to_split.onnx")
    print("Input shape:", input_data.shape)
    print("Output 0 shape:", outputs[0].shape)
    print("Output 1 shape:", outputs[1].shape)
    print("Output 0:")
    print(outputs[0])
    print("Output 1:")
    print(outputs[1])


if __name__ == "__main__":
    main()
