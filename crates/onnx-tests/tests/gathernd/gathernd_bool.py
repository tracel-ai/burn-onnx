#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "numpy",
# ]
# ///

# used to generate model: gathernd_bool.onnx
#
# GatherND on bool data. `output` gathers single elements (index length = data rank), with a
# negative index; `output_slices` gathers rows of 3D data (index length 1 < rank) with
# batch_dims=1. Expected values from onnx.reference.ReferenceEvaluator.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    data = helper.make_tensor_value_info("data", TensorProto.BOOL, [2, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [3])
    data_3d = helper.make_tensor_value_info("data_3d", TensorProto.BOOL, [2, 2, 3])
    indices_3d = helper.make_tensor_value_info("indices_3d", TensorProto.INT64, [2, 1])
    output_slices = helper.make_tensor_value_info("output_slices", TensorProto.BOOL, [2, 3])
    nodes = [
        helper.make_node("GatherND", ["data", "indices"], ["output"]),
        helper.make_node("GatherND", ["data_3d", "indices_3d"], ["output_slices"], batch_dims=1),
    ]
    graph = helper.make_graph(
        nodes, "gathernd_bool_graph", [data, indices, data_3d, indices_3d], [output, output_slices]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "gathernd_bool.onnx")

    feed = {
        "data": np.array([[True, False, True], [False, True, False]]),
        "indices": np.array([[0, 1], [1, 1], [-1, 0]], dtype=np.int64),
        "data_3d": np.array(
            [[[True, False, True], [False, False, True]], [[False, True, True], [True, True, False]]]
        ),
        "indices_3d": np.array([[1], [0]], dtype=np.int64),
    }
    result, slices = ReferenceEvaluator(model).run(None, feed)
    print("output", result.tolist())
    print("output_slices", slices.tolist())


if __name__ == "__main__":
    main()
