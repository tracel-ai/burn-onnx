#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "numpy==2.2.6",
#   "onnx==1.19.0",
# ]
# ///

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parent

CORNER_BOXES = np.array(
    [
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0],
        ]
    ],
    dtype=np.float32,
)
CENTER_BOXES = np.array(
    [
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.6, 0.5, 1.0, 1.0],
            [0.4, 0.5, 1.0, 1.0],
            [10.5, 0.5, 1.0, 1.0],
            [10.6, 0.5, 1.0, 1.0],
            [100.5, 0.5, 1.0, 1.0],
        ]
    ],
    dtype=np.float32,
)
SCORES = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)
MAX_OUTPUT = np.array([3], dtype=np.int64)
IOU_THRESHOLD = np.array([0.5], dtype=np.float32)
SCORE_THRESHOLD = np.array([0.0], dtype=np.float32)


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


def evaluate_and_print(name: str, model, feeds):
    for input_name, value in feeds.items():
        print(f"{name} {input_name}: {value.tolist()}")

    output = ReferenceEvaluator(model).run(None, feeds)[0]
    print(f"{name} selected_indices: {output.tolist()}")
    return output


def save_and_check(name: str, model, feeds):
    onnx.save(model, ROOT / f"{name}.onnx")
    if feeds is not None:
        evaluate_and_print(name, model, feeds)


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

    standard = build_model(
        "non_max_suppression",
        None,
        full_names,
        full_inputs,
        opset=10,
    )
    save_and_check(
        "non_max_suppression",
        standard,
        {
            "boxes": CORNER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    evaluate_and_print(
        "non_max_suppression_score_equal_to_threshold",
        standard,
        {
            "boxes": np.array(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [10.0, 10.0, 11.0, 11.0],
                        [20.0, 20.0, 21.0, 21.0],
                        [30.0, 30.0, 31.0, 31.0],
                        [40.0, 40.0, 41.0, 41.0],
                        [50.0, 50.0, 51.0, 51.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "scores": np.array(
                [[[0.5, 0.6, 0.4, 0.3, 0.2, 0.1]]],
                dtype=np.float32,
            ),
            "max_output_boxes_per_class": np.array([3], dtype=np.int64),
            "iou_threshold": np.array([0.0], dtype=np.float32),
            "score_threshold": np.array([0.5], dtype=np.float32),
        },
    )

    exact_iou = np.float32(0.25 / 1.75)
    evaluate_and_print(
        "non_max_suppression_iou_equal_to_threshold",
        standard,
        {
            "boxes": np.array(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.5, 0.5, 1.5, 1.5],
                        [10.0, 10.0, 11.0, 11.0],
                        [20.0, 20.0, 21.0, 21.0],
                        [30.0, 30.0, 31.0, 31.0],
                        [40.0, 40.0, 41.0, 41.0],
                    ]
                ],
                dtype=np.float32,
            ),
            "scores": np.array(
                [[[0.9, 0.8, -0.1, -0.2, -0.3, -0.4]]],
                dtype=np.float32,
            ),
            "max_output_boxes_per_class": np.array([3], dtype=np.int64),
            "iou_threshold": np.array([exact_iou], dtype=np.float32),
            "score_threshold": np.array([0.0], dtype=np.float32),
        },
    )

    save_and_check(
        "non_max_suppression_center",
        build_model("non_max_suppression_center", 1, full_names, full_inputs),
        {
            "boxes": CENTER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    missing_middle_inputs = [
        value_info("boxes", TensorProto.FLOAT, (1, 6, 4)),
        value_info("scores", TensorProto.FLOAT, (1, 1, 6)),
        value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    missing_middle = build_model(
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
    )
    onnx.save(missing_middle, ROOT / "non_max_suppression_missing_middle.onnx")

    # ONNX 1.19's ReferenceEvaluator doesn't handle an omitted-middle input.
    # Evaluate an equivalent in-memory model with the default IoU threshold.
    reference = build_model(
        "non_max_suppression_missing_middle_reference",
        0,
        full_names,
        full_inputs,
    )
    evaluate_and_print(
        "non_max_suppression_missing_middle",
        reference,
        {
            "boxes": CORNER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": np.array([0.0], dtype=np.float32),
            "score_threshold": np.array([0.8], dtype=np.float32),
        },
    )

    two_box_inputs = standard_inputs(2)
    missing_score_inputs = two_box_inputs[:4]
    two_boxes = CORNER_BOXES[:, :2, :]
    negative_scores = np.array([[[-0.1, -0.2]]], dtype=np.float32)
    save_and_check(
        "non_max_suppression_missing_score_threshold",
        build_model(
            "non_max_suppression_missing_score_threshold",
            0,
            full_names[:4],
            missing_score_inputs,
        ),
        {
            "boxes": two_boxes,
            "scores": negative_scores,
            "max_output_boxes_per_class": np.array([1], dtype=np.int64),
            "iou_threshold": IOU_THRESHOLD,
        },
    )

    minimal_inputs = two_box_inputs[:2]
    minimal = build_model(
        "non_max_suppression_minimal",
        0,
        full_names[:2],
        minimal_inputs,
    )
    save_and_check("non_max_suppression_minimal", minimal, None)

    # The reference evaluator has the same omitted-input bug for this valid
    # minimal model. Evaluate an equivalent model with all defaults explicit.
    minimal_reference = build_model(
        "non_max_suppression_minimal_reference",
        0,
        full_names,
        two_box_inputs,
    )
    evaluate_and_print(
        "non_max_suppression_minimal",
        minimal_reference,
        {
            "boxes": two_boxes,
            "scores": np.array([[[0.9, 0.8]]], dtype=np.float32),
            "max_output_boxes_per_class": np.array([0], dtype=np.int64),
            "iou_threshold": np.array([0.0], dtype=np.float32),
            "score_threshold": np.array([0.0], dtype=np.float32),
        },
    )

    multi_boxes = CORNER_BOXES[:, [0, 1, 3, 4], :]
    multi_scores = np.array(
        [[[0.9, 0.8, 0.7, 0.6], [0.5, 0.6, 0.9, 0.8]]],
        dtype=np.float32,
    )
    multi_inputs = standard_inputs(4)
    multi_inputs[1] = value_info("scores", TensorProto.FLOAT, (1, 2, 4))
    save_and_check(
        "non_max_suppression_multi_class",
        build_model(
            "non_max_suppression_multi_class",
            0,
            full_names,
            multi_inputs,
        ),
        {
            "boxes": multi_boxes,
            "scores": multi_scores,
            "max_output_boxes_per_class": np.array([2], dtype=np.int64),
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )


if __name__ == "__main__":
    main()
