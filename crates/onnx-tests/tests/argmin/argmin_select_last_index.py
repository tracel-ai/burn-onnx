#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/argmin/argmin_select_last_index.onnx
# Exercises the ONNX `select_last_index=1` attribute, which asks for the index
# of the LAST occurrence of the minimum along the axis (not the first).

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT64, [2, 1]
    )

    argmin_node = helper.make_node(
        "ArgMin",
        inputs=["input"],
        outputs=["output"],
        axis=1,
        keepdims=1,
        select_last_index=1,
    )

    graph_def = helper.make_graph(
        [argmin_node],
        "argmin_select_last_index_test",
        [input_tensor],
        [output_tensor],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    onnx_name = "argmin_select_last_index.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    # Input with ties so select_last_index actually matters.
    test_input = np.array(
        [
            [5.0, 1.0, 1.0, 4.0],  # min is 1 at indices 1 and 2; last = 2
            [2.0, 2.0, 2.0, 2.0],  # all tied; last = 3
        ],
        dtype=np.float32,
    )
    print("Test input:\n{}".format(test_input))

    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input": test_input})
    print("Test output:\n{}".format(outputs[0]))


if __name__ == "__main__":
    main()
