#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
Generate OneHotEncoder ONNX models covering every codegen arm:
float32/float64 inputs (cast to integers, per the spec), int64/int32 inputs,
and a 2D input to exercise the category-tensor reshape for rank > 1.

Categories are non-identity (value != column index) and inputs include an
out-of-vocabulary value so an all-zero row is exercised.
Expected outputs are printed from onnx.reference.ReferenceEvaluator.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def make_model(
    name: str, input_dtype: int, input_shape: list, cats: list
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", input_dtype, input_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [*input_shape, len(cats)])

    node = helper.make_node(
        "OneHotEncoder",
        inputs=["X"],
        outputs=["Y"],
        domain="ai.onnx.ml",
        cats_int64s=cats,
        zeros=1,
    )

    graph = helper.make_graph([node], name, [x], [y])
    return helper.make_model(
        graph,
        producer_name="one_hot_encoder-test",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )


def emit(name: str, model: onnx.ModelProto, x: np.ndarray) -> None:
    onnx.save(model, f"{name}.onnx")
    (y,) = ReferenceEvaluator(model).run(None, {"X": x})
    print(f"{name} expected:")
    print(y.tolist())


def main() -> None:
    emit(
        "one_hot_encoder_f32",
        make_model("one_hot_encoder_f32", TensorProto.FLOAT, [None], [1, 2, 4]),
        np.array([1.0, 4.0, 2.0, 1.0], dtype=np.float32),
    )

    # 3.0 is out of vocabulary -> all-zero row.
    emit(
        "one_hot_encoder_f64",
        make_model("one_hot_encoder_f64", TensorProto.DOUBLE, [None], [1, 2, 4]),
        np.array([4.0, 2.0, 3.0, 1.0], dtype=np.float64),
    )

    # Categories above 2^24 are not exact in f32; 16777217 must not match
    # 16777216.0. 5.0 is out of vocabulary.
    emit(
        "one_hot_encoder_f32_large_cats",
        make_model(
            "one_hot_encoder_f32_large_cats",
            TensorProto.FLOAT,
            [None],
            [16777216, 16777217],
        ),
        np.array([16777216.0, 5.0], dtype=np.float32),
    )

    # Integer input path, non-identity categories, 7 is out of vocabulary.
    emit(
        "one_hot_encoder_i64",
        make_model("one_hot_encoder_i64", TensorProto.INT64, [None], [10, 20, 30]),
        np.array([30, 10, 7, 20], dtype=np.int64),
    )

    # 2D int32 input: category tensor reshapes to [1, 1, num_cats].
    emit(
        "one_hot_encoder_2d",
        make_model("one_hot_encoder_2d", TensorProto.INT32, [None, 2], [1, 2, 4]),
        np.array([[1, 4], [2, 9]], dtype=np.int32),
    )


if __name__ == "__main__":
    main()
