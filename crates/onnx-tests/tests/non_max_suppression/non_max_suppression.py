#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent


def value_info(name: str, elem_type: int, shape):
    return helper.make_tensor_value_info(name, elem_type=elem_type, shape=shape)


def save_model(
    name: str,
    node_inputs: list[str],
    graph_inputs,
    opset: int,
):
    node = helper.make_node(
        "NonMaxSuppression",
        inputs=node_inputs,
        outputs=["selected_indices"],
        name=f"/{name}",
    )
    model = helper.make_model(
        graph=helper.make_graph(
            nodes=[node],
            name=name,
            inputs=graph_inputs,
            outputs=[value_info("selected_indices", TensorProto.INT64, ("num_selected", 3))],
        ),
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, ROOT / f"{name}.onnx")


def main():
    full_inputs = [
        value_info("boxes", TensorProto.FLOAT, (1, 2, 4)),
        value_info("scores", TensorProto.FLOAT, (1, 1, 2)),
        value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        value_info("iou_threshold", TensorProto.FLOAT, (1,)),
        value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    full_names = [value.name for value in full_inputs]
    cases = [
        ("non_max_suppression", full_names, full_inputs, 10),
        (
            "non_max_suppression_missing_middle",
            [*full_names[:3], "", full_names[4]],
            [*full_inputs[:3], full_inputs[4]],
            11,
        ),
        ("non_max_suppression_missing_score_threshold", full_names[:4], full_inputs[:4], 11),
        ("non_max_suppression_minimal", full_names[:2], full_inputs[:2], 11),
    ]
    for name, node_inputs, graph_inputs, opset in cases:
        save_model(name, node_inputs, graph_inputs, opset)


if __name__ == "__main__":
    main()
