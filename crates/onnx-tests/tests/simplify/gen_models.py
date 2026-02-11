#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates purpose-built ONNX models that exercise specific simplification
# patterns. Each model is designed so that simplified and unsimplified versions
# produce identical outputs, but the simplified version folds shape computations
# into constants at codegen time.

import onnx
from onnx import helper, TensorProto

OPSET = 16


def save(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, name)
    print(f"  {name}")


def shape_folding():
    """Shape(x) where x has fully static shape -> foldable to constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_out"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "shape_out",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_shape_folding.onnx",
    )


def gather_on_shape():
    """Gather(Shape(x), const_idx) -> single dimension extracted as scalar."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["idx"],
                value=helper.make_tensor("idx_val", TensorProto.INT64, [], [1]),
            ),
            helper.make_node("Gather", ["shape1", "idx"], ["dim_val"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "dim_val",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_gather_on_shape.onnx",
    )


def slice_on_shape():
    """Slice(Shape(x), [1], [3]) -> sub-array of shape dims."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("s", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor("e", TensorProto.INT64, [1], [3]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["axes"],
                value=helper.make_tensor("a", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Slice", ["shape1", "starts", "ends", "axes"], ["sliced"]
            ),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4, 5]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "sliced",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[2]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_slice_on_shape.onnx",
    )


def concat_shapes():
    """Concat(Shape(x), Shape(y)) -> concatenated shape array."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            helper.make_node("Concat", ["shape_x", "shape_y"], ["combined"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[4, 5, 6]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "combined",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[5]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_concat_shapes.onnx",
    )


def reshape_from_shape():
    """Reshape(x, Shape(y)) -> reshape using folded shape constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["shape_source"], ["target_shape"]),
            helper.make_node("Reshape", ["x", "target_shape"], ["reshaped"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[12]),
            ),
            helper.make_value_info(
                "shape_source",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "reshaped",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 4]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_reshape_from_shape.onnx",
    )


def binary_ops_on_shape():
    """Add(Shape(x), const) and Mul(Shape(x), const) on shape values."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "Constant",
                [],
                ["offset"],
                value=helper.make_tensor("off", TensorProto.INT64, [3], [10, 20, 30]),
            ),
            helper.make_node("Add", ["shape1", "offset"], ["added"]),
            helper.make_node(
                "Constant",
                [],
                ["scale"],
                value=helper.make_tensor("sc", TensorProto.INT64, [3], [2, 2, 2]),
            ),
            helper.make_node("Mul", ["shape1", "scale"], ["mulled"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "added",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
            helper.make_value_info(
                "mulled",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_binary_ops_on_shape.onnx",
    )


def cast_shape():
    """Cast(Shape(x), to=FLOAT) -> cast shape to float tensor."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node("Cast", ["shape1"], ["casted"], to=TensorProto.FLOAT),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "casted",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_cast_shape.onnx",
    )


def where_on_shapes():
    """Where(scalar_cond, Shape(x), Shape(y)) -> conditional shape selection.

    Uses scalar bool condition (the only supported pattern for Where with Shape
    outputs besides element-wise Shape,Shape,Shape).
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            helper.make_node("Where", ["cond", "shape_x", "shape_y"], ["selected"]),
        ],
        inputs=[
            helper.make_value_info(
                "cond",
                helper.make_tensor_type_proto(TensorProto.BOOL, shape=[]),
            ),
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[5, 6, 7]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "selected",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_where_on_shapes.onnx",
    )


def expand_from_shape():
    """Expand(x, Shape(y)) -> expand using folded shape constant."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["shape_source"], ["target_shape"]),
            helper.make_node("Expand", ["x", "target_shape"], ["expanded"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 1, 4]),
            ),
            helper.make_value_info(
                "shape_source",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "expanded",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_expand_from_shape.onnx",
    )


def constant_of_shape_opt():
    """ConstantOfShape(Shape(x)) -> constant tensor with known shape."""
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node(
                "ConstantOfShape",
                ["shape1"],
                ["filled"],
                value=helper.make_tensor("val", TensorProto.FLOAT, [1], [5.0]),
            ),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "filled",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_constant_of_shape_opt.onnx",
    )


def gather_shape_chain():
    """Chained Shape->Gather where first Gather's result feeds second Gather's index.

    Shape(x) -> Gather(const 1) -> first_dim (== 1 for x.shape=[3,1,4])
    Shape(y) -> Gather(first_dim) -> second_dim (== dim_1 of y)

    The second Gather uses the first's output as its index. Without proper
    value_store propagation, the second Gather can never be folded because
    its index input has value_source=Constant but value_store=None.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node(
                "Constant",
                [],
                ["idx1"],
                value=helper.make_tensor("idx1_val", TensorProto.INT64, [], [1]),
            ),
            # Gather dim 1 from x's shape: x.shape=[3,1,4] -> 1
            helper.make_node("Gather", ["shape_x", "idx1"], ["dim_from_x"], axis=0),
            helper.make_node("Shape", ["y"], ["shape_y"]),
            # Use the gathered value (1) as index into y's shape: y.shape=[5,6,7] -> 6
            helper.make_node("Gather", ["shape_y", "dim_from_x"], ["dim_from_y"], axis=0),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[3, 1, 4]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[5, 6, 7]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "dim_from_y",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_gather_shape_chain.onnx",
    )


def permute_via_shape_gather():
    """Shape->Gather->Unsqueeze->Concat->Reshape that transposes last two dims.

    input: [2,3,4,5]
    4x Shape -> Gather(0), Gather(1), Gather(3), Gather(2)
    -> Unsqueeze each -> Concat -> [2,3,5,4] -> Reshape(input, shape)
    output: [2,3,5,4]

    This is the classic permute-reshape pattern that should be simplified
    to a Transpose operation.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["input"], ["s0"]),
            helper.make_node("Shape", ["input"], ["s1"]),
            helper.make_node("Shape", ["input"], ["s2"]),
            helper.make_node("Shape", ["input"], ["s3"]),
            helper.make_node(
                "Constant", [], ["i0"],
                value=helper.make_tensor("i0v", TensorProto.INT64, [], [0]),
            ),
            helper.make_node(
                "Constant", [], ["i1"],
                value=helper.make_tensor("i1v", TensorProto.INT64, [], [1]),
            ),
            helper.make_node(
                "Constant", [], ["i3"],
                value=helper.make_tensor("i3v", TensorProto.INT64, [], [3]),
            ),
            helper.make_node(
                "Constant", [], ["i2"],
                value=helper.make_tensor("i2v", TensorProto.INT64, [], [2]),
            ),
            helper.make_node("Gather", ["s0", "i0"], ["d0"], axis=0),
            helper.make_node("Gather", ["s1", "i1"], ["d1"], axis=0),
            helper.make_node("Gather", ["s2", "i3"], ["d3"], axis=0),
            helper.make_node("Gather", ["s3", "i2"], ["d2"], axis=0),
            helper.make_node(
                "Constant", [], ["unsq_axes"],
                value=helper.make_tensor("ax", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node("Unsqueeze", ["d0", "unsq_axes"], ["u0"]),
            helper.make_node("Unsqueeze", ["d1", "unsq_axes"], ["u1"]),
            helper.make_node("Unsqueeze", ["d3", "unsq_axes"], ["u3"]),
            helper.make_node("Unsqueeze", ["d2", "unsq_axes"], ["u2"]),
            helper.make_node("Concat", ["u0", "u1", "u3", "u2"], ["new_shape"], axis=0),
            helper.make_node("Reshape", ["input", "new_shape"], ["output"]),
        ],
        inputs=[
            helper.make_value_info(
                "input",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4, 5]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "output",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 5, 4]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_permute_via_shape_gather.onnx",
    )


def sdpa_coalesce():
    """Decomposed scaled dot-product attention pattern.

    Transpose(K) -> MatMul(Q,K^T) -> Div(sqrt_dk) -> Softmax(-1) -> MatMul(scores,V)

    Should be coalesced into a single Attention node by simplification.
    """
    import math

    batch, heads, seq_len, head_dim = 1, 2, 3, 4
    scale_value = math.sqrt(float(head_dim))

    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            # Transpose K: swap last two dims [b,h,s,d] -> [b,h,d,s]
            helper.make_node(
                "Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]
            ),
            # QK^T matmul: [b,h,s,d] x [b,h,d,s] -> [b,h,s,s]
            helper.make_node("MatMul", ["q", "k_t"], ["qk"]),
            # Scale by 1/sqrt(head_dim)
            helper.make_node(
                "Constant",
                [],
                ["scale"],
                value=helper.make_tensor(
                    "scale_val", TensorProto.FLOAT, [], [scale_value]
                ),
            ),
            helper.make_node("Div", ["qk", "scale"], ["qk_scaled"]),
            # Softmax on last axis
            helper.make_node("Softmax", ["qk_scaled"], ["attn_weights"], axis=-1),
            # Scores x V: [b,h,s,s] x [b,h,s,d] -> [b,h,s,d]
            helper.make_node("MatMul", ["attn_weights", "v"], ["output"]),
        ],
        inputs=[
            helper.make_value_info(
                "q",
                helper.make_tensor_type_proto(
                    TensorProto.FLOAT,
                    shape=[batch, heads, seq_len, head_dim],
                ),
            ),
            helper.make_value_info(
                "k",
                helper.make_tensor_type_proto(
                    TensorProto.FLOAT,
                    shape=[batch, heads, seq_len, head_dim],
                ),
            ),
            helper.make_value_info(
                "v",
                helper.make_tensor_type_proto(
                    TensorProto.FLOAT,
                    shape=[batch, heads, seq_len, head_dim],
                ),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "output",
                helper.make_tensor_type_proto(
                    TensorProto.FLOAT,
                    shape=[batch, heads, seq_len, head_dim],
                ),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_sdpa_coalesce.onnx",
    )


def constant_fold():
    """Mul(Shape->Gather(dim1), Shape->Gather(dim2)) -> constant product.

    x: [2, 3, 4]
    dim1 = Shape(x) -> Gather(idx=1) -> 3   (folded by constant_shape)
    dim2 = Shape(x) -> Gather(idx=2) -> 4   (folded by constant_shape)
    product = Mul(dim1, dim2) -> 12          (folded by constant_fold)
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape1"]),
            helper.make_node("Shape", ["x"], ["shape2"]),
            helper.make_node(
                "Constant",
                [],
                ["idx1"],
                value=helper.make_tensor("idx1_val", TensorProto.INT64, [], [1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["idx2"],
                value=helper.make_tensor("idx2_val", TensorProto.INT64, [], [2]),
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
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_constant_fold.onnx",
    )


if __name__ == "__main__":
    print("Generating simplify test models:")
    shape_folding()
    gather_on_shape()
    slice_on_shape()
    concat_shapes()
    reshape_from_shape()
    binary_ops_on_shape()
    cast_shape()
    where_on_shapes()
    expand_from_shape()
    constant_of_shape_opt()
    gather_shape_chain()
    permute_via_shape_gather()
    sdpa_coalesce()
    constant_fold()
    print("Done.")
