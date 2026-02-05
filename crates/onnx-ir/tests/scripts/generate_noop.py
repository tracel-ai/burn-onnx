#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
# ]
# ///

"""
Generate ONNX models testing no-op node elimination.

Tests that nodes which are effectively no-ops get eliminated during
post-processing via the is_noop() processor trait method:
- Cast with same input/output dtype (F32 -> F32)
- Identity nodes (always no-ops)
"""

import onnx
from onnx import helper, TensorProto


def create_cast_same_type_model():
    """Create model with Cast(F32 -> F32) which is a no-op."""

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    # input -> cast(F32->F32) -> relu -> output
    # The Cast should be eliminated as a no-op
    nodes = [
        helper.make_node("Cast", ["input"], ["cast_out"], name="cast", to=TensorProto.FLOAT),
        helper.make_node("Relu", ["cast_out"], ["output"], name="relu"),
    ]

    graph = helper.make_graph(nodes, "cast_noop_model", [input_tensor], [output])
    model = helper.make_model(
        graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)]
    )
    onnx.checker.check_model(model)
    return model


def create_cast_different_type_model():
    """Create model with Cast(F32 -> I64) which is NOT a no-op."""

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [2, 3])

    # input -> cast(F32->I64) -> output
    # The Cast should NOT be eliminated
    nodes = [
        helper.make_node("Cast", ["input"], ["output"], name="cast", to=TensorProto.INT64),
    ]

    graph = helper.make_graph(nodes, "cast_not_noop_model", [input_tensor], [output])
    model = helper.make_model(
        graph, producer_name="onnx-ir-test", opset_imports=[helper.make_opsetid("", 16)]
    )
    onnx.checker.check_model(model)
    return model


def main():
    """Generate and save all no-op test models."""
    models = {
        "noop_cast_same_type.onnx": create_cast_same_type_model(),
        "noop_cast_different_type.onnx": create_cast_different_type_model(),
    }

    for filename, model in models.items():
        output_path = f"../fixtures/{filename}"
        onnx.save(model, output_path)
        print(f"Saved {output_path}")
        print(f"  Nodes: {len(model.graph.node)}")
        for node in model.graph.node:
            print(f"    - {node.op_type} ({node.name}): {list(node.input)} -> {list(node.output)}")


if __name__ == "__main__":
    main()
