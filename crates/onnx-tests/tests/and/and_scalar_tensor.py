#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Regression test: And with a scalar bool constant and a bool tensor input.
# This pattern occurs in ALBERT where a constant bool is AND-ed with a mask tensor.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    # Constant: scalar true
    const_true = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_true"],
        value=helper.make_tensor("val", TensorProto.BOOL, [], [True]),
    )

    # And(const_true, input) -> output should equal input
    and_node = helper.make_node(
        "And", ["const_true", "input"], ["output"]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.BOOL, [2, 3])
    output_info = helper.make_tensor_value_info("output", TensorProto.BOOL, [2, 3])

    graph = helper.make_graph(
        [const_true, and_node],
        "and_scalar_tensor",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.save(model, "and_scalar_tensor.onnx")
    print("Saved and_scalar_tensor.onnx")

    # Verify with ReferenceEvaluator
    test_input = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    session = ReferenceEvaluator("and_scalar_tensor.onnx")
    [result] = session.run(None, {"input": test_input})
    print(f"Input:  {test_input}")
    print(f"Output: {result}")
    assert np.array_equal(result, test_input), "true AND x should equal x"


if __name__ == "__main__":
    main()
