#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
# ]
# ///

"""
Generate ONNX models for testing the constant folding simplification pass.

Models exercise various constant folding scenarios at the IR level:
1. Cascading shape->gather->mul fold
2. Chained arithmetic (add then sub on constants)
3. Dynamic input blocking fold (negative test)
"""

import onnx
from onnx import helper, TensorProto


def save(model, name):
    onnx.checker.check_model(model)
    path = f"../fixtures/{name}"
    onnx.save(model, path)
    print(f"  {path}")


def constant_fold_cascade():
    """Shape->Gather(dim1) * Shape->Gather(dim2) folds to a single constant.

    x: [2, 3, 4]
    dim1 = Shape(x)[1] = 3   (folded by constant_shape)
    dim2 = Shape(x)[2] = 4   (folded by constant_shape)
    product = Mul(dim1, dim2) = 12  (folded by constant_fold)
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node("Shape", ["x"], ["shape2"]),
            helper.make_node(
                "Constant", [], ["idx1"],
                value=helper.make_tensor("v", TensorProto.INT64, [], [1]),
            ),
            helper.make_node(
                "Constant", [], ["idx2"],
                value=helper.make_tensor("v", TensorProto.INT64, [], [2]),
            ),
            helper.make_node("Gather", ["shape1", "idx1"], ["dim1"], axis=0),
            helper.make_node("Gather", ["shape2", "idx2"], ["dim2"], axis=0),
            helper.make_node("Mul", ["dim1", "dim2"], ["product"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "product",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_cascade.onnx",
    )


def constant_fold_chain():
    """Chained arithmetic on constants: Add(2, 3) -> Sub(result, 1) -> 4.

    All inputs are initializer constants, so the entire chain should fold.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Add", ["a", "b"], ["sum"]),
            helper.make_node("Sub", ["sum", "c"], ["result"]),
        ],
        inputs=[],
        outputs=[
            helper.make_value_info(
                "result",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
        initializer=[
            helper.make_tensor("a", TensorProto.INT64, [], [2]),
            helper.make_tensor("b", TensorProto.INT64, [], [3]),
            helper.make_tensor("c", TensorProto.INT64, [], [1]),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_chain.onnx",
    )


def constant_fold_blocked():
    """Dynamic input blocks folding: Add(dynamic_x, const) cannot fold.

    Only the Neg(const) can be folded; the Add must remain.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            # Neg(-5) = 5, foldable
            helper.make_node("Neg", ["offset"], ["neg_offset"]),
            # Add(x, 5), NOT foldable (x is dynamic)
            helper.make_node("Add", ["x", "neg_offset"], ["result"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "result",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        initializer=[
            helper.make_tensor("offset", TensorProto.FLOAT, [], [-5.0]),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_blocked.onnx",
    )


def constant_fold_concat():
    """Concat of constant arrays folds into a single constant.

    Shape(x)[0:2] ++ Shape(y)[0:1] -> [2, 3, 4]
    Uses Shape->Slice to produce constant arrays, then Concat merges them.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            helper.make_node(
                "Constant", [], ["starts_a"],
                value=helper.make_tensor("v", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Constant", [], ["ends_a"],
                value=helper.make_tensor("v", TensorProto.INT64, [1], [2]),
            ),
            helper.make_node(
                "Constant", [], ["starts_b"],
                value=helper.make_tensor("v", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Constant", [], ["ends_b"],
                value=helper.make_tensor("v", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node("Slice", ["shape_x", "starts_a", "ends_a"], ["slice_x"]),
            helper.make_node("Slice", ["shape_y", "starts_b", "ends_b"], ["slice_y"]),
            helper.make_node("Concat", ["slice_x", "slice_y"], ["result"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[5, 6]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "result",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_concat.onnx",
    )


def constant_fold_cast():
    """Shape->Gather folds to i64 constant, then Cast(to=FLOAT) folds to f32.

    x: [2, 3, 4]
    dim = Shape(x)[1] = 3   (folded by constant_shape)
    float_dim = Cast(dim, to=FLOAT) = 3.0  (folded by constant_fold)
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant", [], ["idx"],
                value=helper.make_tensor("v", TensorProto.INT64, [], [1]),
            ),
            helper.make_node("Gather", ["shape1", "idx"], ["dim"], axis=0),
            helper.make_node("Cast", ["dim"], ["float_dim"], to=TensorProto.FLOAT),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "float_dim",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_cast.onnx",
    )


def constant_fold_sqrt():
    """Shape->Gather->Cast->Sqrt cascade: computes sqrt(head_dim) at compile time.

    x: [2, 3, 64]
    dim = Shape(x)[2] = 64       (folded by constant_shape)
    float_dim = Cast(dim) = 64.0  (folded by constant_fold: Cast)
    scale = Sqrt(float_dim) = 8.0 (folded by constant_fold: Sqrt)
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant", [], ["idx"],
                value=helper.make_tensor("v", TensorProto.INT64, [], [2]),
            ),
            helper.make_node("Gather", ["shape1", "idx"], ["dim"], axis=0),
            helper.make_node("Cast", ["dim"], ["float_dim"], to=TensorProto.FLOAT),
            helper.make_node("Sqrt", ["float_dim"], ["scale"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 64]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "scale",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)]),
        "constant_fold_sqrt.onnx",
    )


if __name__ == "__main__":
    print("Generating constant fold test models:")
    constant_fold_cascade()
    constant_fold_chain()
    constant_fold_blocked()
    constant_fold_concat()
    constant_fold_cast()
    constant_fold_sqrt()
    print("Done.")
