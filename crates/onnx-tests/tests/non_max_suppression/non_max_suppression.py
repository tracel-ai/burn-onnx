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


def build_model(
    name: str,
    center_point_box: int | None,
    node_inputs: list[str],
    graph_inputs,
    opset: int = 11,
):
    attributes = {}
    if center_point_box is not None:
        attributes["center_point_box"] = center_point_box

    node = helper.make_node(
        "NonMaxSuppression",
        inputs=node_inputs,
        outputs=["selected_indices"],
        name=f"/{name}",
        **attributes,
    )
    model = helper.make_model(
        graph=helper.make_graph(
            nodes=[node],
            name=name,
            inputs=graph_inputs,
            outputs=[
                value_info(
                    "selected_indices",
                    TensorProto.INT64,
                    ("num_selected", 3),
                )
            ],
        ),
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return model


def save_model(name: str, model):
    onnx.save(model, ROOT / f"{name}.onnx")


def standard_inputs(num_boxes: int = 6):
    return [
        value_info("boxes", TensorProto.FLOAT, (1, num_boxes, 4)),
        value_info("scores", TensorProto.FLOAT, (1, 1, num_boxes)),
        value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        value_info("iou_threshold", TensorProto.FLOAT, (1,)),
        value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]


def main():
    full_inputs = standard_inputs()
    full_names = [
        "boxes",
        "scores",
        "max_output_boxes_per_class",
        "iou_threshold",
        "score_threshold",
    ]
    save_model(
        "non_max_suppression",
        build_model(
            "non_max_suppression",
            None,
            full_names,
            full_inputs,
            opset=10,
        ),
    )

    missing_middle_inputs = [
        value_info("boxes", TensorProto.FLOAT, (1, 6, 4)),
        value_info("scores", TensorProto.FLOAT, (1, 1, 6)),
        value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    save_model(
        "non_max_suppression_missing_middle",
        build_model(
            "non_max_suppression_missing_middle",
            0,
            [
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "",
                "score_threshold",
            ],
            missing_middle_inputs,
        ),
    )

    two_box_inputs = standard_inputs(2)
    save_model(
        "non_max_suppression_missing_score_threshold",
        build_model(
            "non_max_suppression_missing_score_threshold",
            0,
            full_names[:4],
            two_box_inputs[:4],
        ),
    )
    save_model(
        "non_max_suppression_minimal",
        build_model(
            "non_max_suppression_minimal",
            0,
            full_names[:2],
            two_box_inputs[:2],
        ),
    )


if __name__ == "__main__":
    main()
