#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# Generates two fixtures for the If/Loop/Scan subgraph limits of custom op
# hooks (see DESIGN-CUSTOM-OPS.md, open question 1):
#
#   custom_in_if.onnx: If whose then-branch contains a custom-domain op
#     (test.custom::SubCustom); the else-branch is a built-in Relu. Used to
#     verify inference hooks reach subgraph bodies and that burn-onnx rejects
#     the body at codegen.
#   relu_in_if.onnx: If whose both branches are built-in Relu. Used to verify
#     burn-onnx rejects OpOverride targets appearing inside subgraph bodies.
#
# Both branches reference the outer-scope value "x".

import onnx
from onnx import TensorProto, helper


def branch(name, node):
    return helper.make_graph(
        [node],
        name,
        inputs=[],
        outputs=[helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [2])],
    )


def if_model(then_node, else_node):
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="if1",
        then_branch=branch("then_branch", then_node),
        else_branch=branch("else_branch", else_node),
    )
    graph = helper.make_graph(
        [if_node],
        "subgraph_custom_graph",
        inputs=[
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[
            helper.make_operatorsetid("", 16),
            helper.make_operatorsetid("test.custom", 1),
        ],
    )


def main():
    custom = helper.make_node(
        "SubCustom",
        inputs=["x"],
        outputs=["branch_out"],
        name="sub_custom1",
        domain="test.custom",
    )
    relu_then = helper.make_node(
        "Relu", inputs=["x"], outputs=["branch_out"], name="relu_then"
    )
    relu_else = helper.make_node(
        "Relu", inputs=["x"], outputs=["branch_out"], name="relu_else"
    )

    onnx.save(if_model(custom, relu_else), "../fixtures/custom_in_if.onnx")
    onnx.save(if_model(relu_then, relu_else), "../fixtures/relu_in_if.onnx")
    print("Finished exporting custom_in_if.onnx and relu_in_if.onnx")


if __name__ == "__main__":
    main()
