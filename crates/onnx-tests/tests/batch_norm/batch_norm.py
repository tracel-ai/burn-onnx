#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: batch_norm.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    init_batch_norm1d_weight = numpy_helper.from_array(
        np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        ).reshape([20]),
        name="batch_norm1d.weight",
    )

    init_batch_norm1d_bias = numpy_helper.from_array(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ).reshape([20]),
        name="batch_norm1d.bias",
    )

    init_batch_norm2d_weight = numpy_helper.from_array(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape([5]),
        name="batch_norm2d.weight",
    )

    init_batch_norm2d_bias = numpy_helper.from_array(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape([5]),
        name="batch_norm2d.bias",
    )

    node0 = helper.make_node(
        "Identity", ["batch_norm2d.weight"], ["batch_norm2d.running_var"]
    )
    node1 = helper.make_node(
        "Identity", ["batch_norm2d.bias"], ["batch_norm2d.running_mean"]
    )
    node2 = helper.make_node(
        "Identity", ["batch_norm1d.weight"], ["batch_norm1d.running_var"]
    )
    node3 = helper.make_node(
        "Identity", ["batch_norm1d.bias"], ["batch_norm1d.running_mean"]
    )
    node4 = helper.make_node(
        "BatchNormalization",
        [
            "input.1",
            "batch_norm1d.weight",
            "batch_norm1d.bias",
            "batch_norm1d.running_mean",
            "batch_norm1d.running_var",
        ],
        ["/batch_norm1d/BatchNormalization_output_0"],
        epsilon=9.999999747378752e-06,
        momentum=0.8999999761581421,
        training_mode=0,
    )
    node5 = helper.make_node(
        "Constant",
        [],
        ["/Constant_output_0"],
        value=numpy_helper.from_array(
            np.array([1, 5, 2, 2], dtype=np.int64).reshape([4]), name="value"
        ),
    )
    node6 = helper.make_node(
        "Reshape",
        ["/batch_norm1d/BatchNormalization_output_0", "/Constant_output_0"],
        ["/Reshape_output_0"],
        allowzero=0,
    )
    node7 = helper.make_node(
        "BatchNormalization",
        [
            "/Reshape_output_0",
            "batch_norm2d.weight",
            "batch_norm2d.bias",
            "batch_norm2d.running_mean",
            "batch_norm2d.running_var",
        ],
        ["14"],
        epsilon=9.999999747378752e-06,
        momentum=0.8999999761581421,
        training_mode=0,
    )

    inp_input_1 = helper.make_tensor_value_info(
        "input.1", TensorProto.FLOAT, [1, 20, 1]
    )

    out_n14 = helper.make_tensor_value_info("14", TensorProto.FLOAT, [1, 5, 2, 2])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5, node6, node7],
        "torch_jit",
        [inp_input_1],
        [out_n14],
        initializer=[
            init_batch_norm1d_weight,
            init_batch_norm1d_bias,
            init_batch_norm2d_weight,
            init_batch_norm2d_bias,
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "batch_norm.onnx")
    print(f"Finished exporting model to batch_norm.onnx")


if __name__ == "__main__":
    main()
