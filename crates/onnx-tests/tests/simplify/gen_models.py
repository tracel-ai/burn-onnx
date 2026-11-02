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


def sdpa_prescale_alias():
    """Decomposed SDPA whose Q pre-scale hides behind an additive identity.

    Q -> Mul(0.25) -> Add(0) -> Identity -\\
                                            MatMul -> Softmax(-1) -> MatMul(V)
    K -> Transpose([0,1,3,2]) ------------/

    Identity is dropped in post-processing. Add(0) survives until the identity
    element pass, which runs after attention coalescing, so the coalescer sees
    Add -> MatMul and cannot reach the Mul. By the time Add(0) is removed the
    Attention node already exists, so the Mul stays as an ordinary node and the
    coalesced Attention must carry an explicit unit scale.

    head_dim is 4, so a wrongly applied 1/sqrt(head_dim) default is 0.5. The Q
    pre-scale is deliberately 0.25 rather than 0.5 so that it does not coincide
    with that default: were they equal, dropping the Mul while leaving the scale
    absent would cancel out and produce a zero output difference.
    """
    batch, heads, seq_len, head_dim = 1, 1, 2, 4

    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node(
                "Constant",
                [],
                ["q_scale"],
                value=helper.make_tensor("q_scale_val", TensorProto.FLOAT, [], [0.25]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["zero"],
                value=helper.make_tensor("zero_val", TensorProto.FLOAT, [], [0.0]),
            ),
            helper.make_node("Mul", ["q", "q_scale"], ["q_scaled"]),
            helper.make_node("Add", ["q_scaled", "zero"], ["q_noop"]),
            helper.make_node("Identity", ["q_noop"], ["q_alias"]),
            helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["q_alias", "k_t"], ["qk"]),
            helper.make_node("Softmax", ["qk"], ["attn_weights"], axis=-1),
            helper.make_node("MatMul", ["attn_weights", "v"], ["output"]),
        ],
        inputs=[
            helper.make_value_info(
                name,
                helper.make_tensor_type_proto(
                    TensorProto.FLOAT,
                    shape=[batch, heads, seq_len, head_dim],
                ),
            )
            for name in ("q", "k", "v")
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
        "simplify_sdpa_prescale_alias.onnx",
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


def expand_shape_chain():
    """Expand target built by torch's `expand` export pattern (issue #280).

    x: [2, 1], y: [2, 3]
    tail = Slice(Shape(y), 2, MAX) -> []       (folded by constant_shape)
    target = Concat([1, 3], tail) -> [1, 3]    (folded by constant_fold)
    ones = ConstantOfShape(Shape(target))      -> [1, 1]
    eq = Equal(target, Mul(ones, -1))          -> [False, False]
    Expand(x, Where(eq, ones, target))         -> [2, 3]

    `target` is inferred as a Shape before folding, and Equal/Where are typed
    against that, so the folded constant must stay a Shape.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["y"], ["y_shape"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("starts_val", TensorProto.INT64, [1], [2]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor(
                    "ends_val", TensorProto.INT64, [1], [9223372036854775807]
                ),
            ),
            helper.make_node("Slice", ["y_shape", "starts", "ends"], ["tail"]),
            helper.make_node(
                "Constant",
                [],
                ["head"],
                value=helper.make_tensor("head_val", TensorProto.INT64, [2], [1, 3]),
            ),
            helper.make_node("Concat", ["head", "tail"], ["target"], axis=0),
            helper.make_node("Shape", ["target"], ["target_len"]),
            helper.make_node(
                "ConstantOfShape",
                ["target_len"],
                ["ones"],
                value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["neg1"],
                value=helper.make_tensor("neg1_val", TensorProto.INT64, [], [-1]),
            ),
            helper.make_node("Mul", ["ones", "neg1"], ["neg_ones"]),
            helper.make_node("Equal", ["target", "neg_ones"], ["eq"]),
            helper.make_node("Where", ["eq", "ones", "target"], ["expand_shape"]),
            helper.make_node("Expand", ["x", "expand_shape"], ["expanded"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 1]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "expanded",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_expand_shape_chain.onnx",
    )


def squeeze_shape_dim():
    """Add(x, Squeeze(Slice(Shape(y), 1, 2))) with a folded scalar dim.

    y: [2, 3]
    dim = Slice(Shape(y), 1, 2) -> [3]   (folded by constant_shape)
    n = Squeeze(dim) -> 3                 (folded by constant_fold)
    out = Add(x, n)

    `n` is inferred as a scalar before folding, and Add is typed against
    that, so the folded constant must stay a scalar.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["y"], ["y_shape"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("starts_val", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor("ends_val", TensorProto.INT64, [1], [2]),
            ),
            helper.make_node("Slice", ["y_shape", "starts", "ends"], ["dim"]),
            helper.make_node(
                "Constant",
                [],
                ["axes"],
                value=helper.make_tensor("axes_val", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node("Squeeze", ["dim", "axes"], ["n"]),
            helper.make_node("Add", ["x", "n"], ["out"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[2]),
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "out",
                helper.make_tensor_type_proto(TensorProto.INT64, shape=[2]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_squeeze_shape_dim.onnx",
    )


def reshape_concat_shape():
    """Reshape(x, Concat(Slice(Shape(x)), [-1])) -> reshape with a lifted static shape.

    x: [2, 3, 4]
    lead = Shape(x)[0:1] -> [2]                 (folded by constant_shape)
    target = Concat(lead, [-1]) -> [2, -1]      (folded by constant_fold)
    reshaped = Reshape(x, target) -> [2, 12]

    The Concat only becomes constant during simplification, after the initial
    constant lifting pass. The Reshape must still lift it into its static
    config so the folded Concat constant is dropped instead of being emitted
    as an unused binding.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("starts_val", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor("ends_val", TensorProto.INT64, [1], [1]),
            ),
            helper.make_node("Slice", ["shape_x", "starts", "ends"], ["lead"]),
            helper.make_node(
                "Constant",
                [],
                ["minus_one"],
                value=helper.make_tensor("minus_one_val", TensorProto.INT64, [1], [-1]),
            ),
            helper.make_node("Concat", ["lead", "minus_one"], ["target"], axis=0),
            helper.make_node("Reshape", ["x", "target"], ["reshaped"]),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 3, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "reshaped",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 12]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_reshape_concat_shape.onnx",
    )


def resize_sizes_from_shape():
    """Resize(x, sizes=Concat(Slice(Shape(x)), [8, 8])) -> resize with static sizes.

    x: [1, 3, 4, 4]
    lead = Shape(x)[0:2] -> [1, 3]              (folded by constant_shape)
    sizes = Concat(lead, [8, 8]) -> [1, 3, 8, 8] (folded to a Shape-typed constant)
    resized = Resize(x, sizes) -> [1, 3, 8, 8]

    Resize lifts the folded sizes constant, so its config must read the value
    of the Shape-typed input rather than treat it as a runtime argument.
    """
    graph = helper.make_graph(
        name="main_graph",
        nodes=[
            helper.make_node("Shape", ["x"], ["shape_x"]),
            helper.make_node(
                "Constant",
                [],
                ["starts"],
                value=helper.make_tensor("starts_val", TensorProto.INT64, [1], [0]),
            ),
            helper.make_node(
                "Constant",
                [],
                ["ends"],
                value=helper.make_tensor("ends_val", TensorProto.INT64, [1], [2]),
            ),
            helper.make_node("Slice", ["shape_x", "starts", "ends"], ["lead"]),
            helper.make_node(
                "Constant",
                [],
                ["spatial"],
                value=helper.make_tensor("spatial_val", TensorProto.INT64, [2], [8, 8]),
            ),
            helper.make_node("Concat", ["lead", "spatial"], ["sizes"], axis=0),
            helper.make_node(
                "Resize", ["x", "", "", "sizes"], ["resized"], mode="nearest",
                coordinate_transformation_mode="asymmetric", nearest_mode="floor"
            ),
        ],
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 3, 4, 4]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "resized",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 3, 8, 8]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_resize_sizes_from_shape.onnx",
    )


def pool_output_dims():
    """Gather(Shape(pool(x)), 2) for MaxPool and LpPool -> constant pooled height.

    x: [1, 3, 7, 7]
    floor:        MaxPool kernel 2, stride 2                    -> 3
    ceil:         MaxPool kernel 2, stride 2, pads 1/1, ceil    -> 4
    ceil_asym:    MaxPool kernel 4, stride 4, pads 0/3, ceil    -> 2
    same:         MaxPool kernel 3, stride 2, SAME_UPPER        -> 4
    lppool:       LpPool kernel 3, stride 2                     -> 3
    lppool_ceil:  LpPool kernel 4, stride 4, pads 0/3, ceil     -> 2

    Each ceil case drops a last window that would start in the end padding
    (5 -> 4 and 3 -> 2). burn's pool modules keep it for asymmetric pads, so
    the generated forward trims it. LpPool's ceil_mode needs opset 18.

    The folded height comes from the pool's static output shape, so it must
    follow the pooling formula rather than repeat the input's 7.
    """
    pools = {
        "floor": ("MaxPool", dict(kernel_shape=[2, 2], strides=[2, 2])),
        "ceil": (
            "MaxPool",
            dict(kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1], ceil_mode=1),
        ),
        "ceil_asym": (
            "MaxPool",
            dict(kernel_shape=[4, 4], strides=[4, 4], pads=[0, 0, 3, 3], ceil_mode=1),
        ),
        "same": (
            "MaxPool",
            dict(kernel_shape=[3, 3], strides=[2, 2], auto_pad="SAME_UPPER"),
        ),
        "lppool": ("LpPool", dict(kernel_shape=[3, 3], strides=[2, 2])),
        "lppool_ceil": (
            "LpPool",
            dict(kernel_shape=[4, 4], strides=[4, 4], pads=[0, 0, 3, 3], ceil_mode=1),
        ),
    }
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["axis"],
            value=helper.make_tensor("axis_val", TensorProto.INT64, [], [2]),
        )
    ]
    outputs = []
    for name, (op, attrs) in pools.items():
        nodes.append(helper.make_node(op, ["x"], [f"y_{name}"], **attrs))
        nodes.append(helper.make_node("Shape", [f"y_{name}"], [f"shape_{name}"]))
        nodes.append(helper.make_node("Gather", [f"shape_{name}", "axis"], [f"h_{name}"]))
        outputs.append(
            helper.make_value_info(
                f"h_{name}", helper.make_tensor_type_proto(TensorProto.INT64, shape=[])
            )
        )
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 3, 7, 7]),
            ),
        ],
        outputs=outputs,
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)]),
        "simplify_pool_output_dims.onnx",
    )


def reshape_symbolic_dims():
    """Relative-position shift with symbolic value_info, as Conformer exports emit it.

    F.pad(x, (1, 0)).view(b, h, -1, t)[:, :, 1:].view(b, h, t, -1)

    x: [b, 2, t, 2t-1]. Both Reshape targets come from Shape(x), and the value_info on each
    side of a Reshape is [sym, 2, sym, sym]. Unknown dims must not count as equal, or both
    Reshapes look like no-ops.
    """

    def const(name, values):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(
                f"{name}_val", TensorProto.INT64, [len(values)], values
            ),
        )

    def scalar(name, value):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(f"{name}_val", TensorProto.INT64, [], [value]),
        )

    def info(name, shape):
        return helper.make_value_info(
            name, helper.make_tensor_type_proto(TensorProto.FLOAT, shape=shape)
        )

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        const("pads", [0, 0, 0, 1, 0, 0, 0, 0]),
        helper.make_node("Pad", ["x", "pads"], ["padded"]),
        scalar("i0", 0),
        scalar("i2", 2),
        scalar("i3", 3),
        const("axes0", [0]),
        helper.make_node("Gather", ["shape", "i0"], ["b"], axis=0),
        helper.make_node("Gather", ["shape", "i2"], ["t"], axis=0),
        helper.make_node("Gather", ["shape", "i3"], ["p"], axis=0),
        helper.make_node("Unsqueeze", ["b", "axes0"], ["b1"]),
        helper.make_node("Unsqueeze", ["t", "axes0"], ["t1"]),
        helper.make_node("Unsqueeze", ["p", "axes0"], ["p1"]),
        const("heads", [2]),
        const("minus_one", [-1]),
        helper.make_node("Concat", ["b1", "heads", "minus_one", "t1"], ["view1"], axis=0),
        helper.make_node("Concat", ["b1", "heads", "t1", "p1"], ["view2"], axis=0),
        helper.make_node("Reshape", ["padded", "view1"], ["r1"]),
        const("starts", [1]),
        const("ends", [2**62]),
        const("axes2", [2]),
        helper.make_node("Slice", ["r1", "starts", "ends", "axes2"], ["sliced"]),
        helper.make_node("Reshape", ["sliced", "view2"], ["y"]),
    ]
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[info("x", ["b", 2, "t", "p"])],
        outputs=[info("y", ["b", 2, "t", "p"])],
        value_info=[
            info("padded", ["b", 2, "t", "p_plus_1"]),
            info("r1", ["b", 2, "two_t", "t"]),
            info("sliced", ["b", 2, "two_t_minus_1", "t"]),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_reshape_symbolic_dims.onnx",
    )


def pad_from_constants():
    """F.pad amounts built from constants only, as PyTorch exports them.

    ConstantOfShape -> Concat -> Reshape -> reversed Slice -> Transpose -> Reshape -> Pad.
    The pads fold to [0, 0, 0, 0, 0, 1] at import, so Pad gets static amounts instead of
    reading them back from the device every call.
    """

    def const(name, values):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(
                f"{name}_val", TensorProto.INT64, [len(values)], values
            ),
        )

    nodes = [
        const("c", [4]),
        helper.make_node(
            "ConstantOfShape",
            ["c"],
            ["zeros"],
            value=helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
        ),
        const("p", [0, 1]),
        helper.make_node("Concat", ["p", "zeros"], ["pads0"], axis=0),
        const("shp", [-1, 2]),
        helper.make_node("Reshape", ["pads0", "shp"], ["r"]),
        const("st", [-1]),
        const("en", [-(2**62)]),
        const("ax", [0]),
        const("stp", [-1]),
        helper.make_node("Slice", ["r", "st", "en", "ax", "stp"], ["rev"]),
        helper.make_node("Transpose", ["rev"], ["tr"], perm=[1, 0]),
        const("flat", [-1]),
        helper.make_node("Reshape", ["tr", "flat"], ["pads"]),
        helper.make_node("Pad", ["x", "pads"], ["y"]),
    ]
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[
            helper.make_value_info(
                "x",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 4, "t"]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 4, "u"]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]),
        "simplify_pad_from_constants.onnx",
    )


def ones_times_minus_one():
    """expand(-1) bookkeeping with a constant length, as RF-DETR exports it.

    ones = ConstantOfShape([3], 1); y = Where(Equal(x, ones * -1), ones, x) turns each -1
    in x into 1. Once ones folds to [1, 1, 1], `ones * -1` must not be treated as `1 * x`:
    the scalar -1 is broadcast to the shape of ones.
    """
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["len"],
            value=helper.make_tensor("len_val", TensorProto.INT64, [1], [3]),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["len"],
            ["ones"],
            value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        ),
        helper.make_node(
            "Constant",
            [],
            ["minus_one"],
            value=helper.make_tensor("minus_one_val", TensorProto.INT64, [], [-1]),
        ),
        helper.make_node("Mul", ["ones", "minus_one"], ["negs"]),
        helper.make_node("Equal", ["x", "negs"], ["is_neg"]),
        helper.make_node("Where", ["is_neg", "ones", "x"], ["y"]),
    ]
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[
            helper.make_value_info(
                "x", helper.make_tensor_type_proto(TensorProto.INT64, shape=[3])
            ),
        ],
        outputs=[
            helper.make_value_info(
                "y", helper.make_tensor_type_proto(TensorProto.INT64, shape=[3])
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_ones_times_minus_one.onnx",
    )


def expand_minus_one_sizes():
    """x.expand(y.size(0), -1, y.size(2)), as the TorchScript exporter emits it.

    The -1 is resolved at runtime with ConstantOfShape(Shape(sizes)), a Mul by -1, an Equal
    and a Where. `sizes` is a Shape(3), so Shape(sizes) is the constant [3] and the whole
    chain stays on the host: Expand gets its shape without a device readback.
    """

    def const(name, values, dims):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(f"{name}_val", TensorProto.INT64, dims, values),
        )

    nodes = [
        helper.make_node("Shape", ["y"], ["y_shape"]),
        const("i0", [0], []),
        helper.make_node("Gather", ["y_shape", "i0"], ["b"], axis=0),
        const("i2", [2], []),
        helper.make_node("Gather", ["y_shape", "i2"], ["t"], axis=0),
        const("ax0", [0], [1]),
        helper.make_node("Unsqueeze", ["b", "ax0"], ["b1"]),
        const("neg1", [-1], [1]),
        helper.make_node("Unsqueeze", ["t", "ax0"], ["t1"]),
        helper.make_node("Concat", ["b1", "neg1", "t1"], ["sizes0"], axis=0),
        const("flat", [-1], [1]),
        helper.make_node("Reshape", ["sizes0", "flat"], ["sizes"]),
        helper.make_node("Shape", ["sizes"], ["sizes_len"]),
        helper.make_node(
            "ConstantOfShape",
            ["sizes_len"],
            ["ones"],
            value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        ),
        const("minus_one", [-1], []),
        helper.make_node("Mul", ["ones", "minus_one"], ["negs"]),
        helper.make_node("Equal", ["sizes", "negs"], ["is_neg"]),
        helper.make_node("Where", ["is_neg", "ones", "sizes"], ["shape"]),
        helper.make_node("Expand", ["x", "shape"], ["z"]),
    ]
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[
            helper.make_value_info(
                "x", helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[1, 4, 1])
            ),
            helper.make_value_info(
                "y",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=["b", 4, "t"]),
            ),
        ],
        outputs=[
            helper.make_value_info(
                "z",
                helper.make_tensor_type_proto(TensorProto.FLOAT, shape=["b", 4, "t"]),
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_expand_minus_one_sizes.onnx",
    )


def ones_cumsum_positions():
    """Position ids as ones(n).cumsum(0) with n a folded static dim.

    The ConstantOfShape length folds to [8], but CumSum and the Add after it need a
    tensor, so the fill must stay a tensor rather than move to the host as a Shape.
    """

    def const(name, values, dims):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(f"{name}_val", TensorProto.INT64, dims, values),
        )

    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        const("i1", [1], []),
        helper.make_node("Gather", ["x_shape", "i1"], ["n"], axis=0),
        const("ax0", [0], [1]),
        helper.make_node("Unsqueeze", ["n", "ax0"], ["n1"]),
        helper.make_node(
            "ConstantOfShape",
            ["n1"],
            ["ones"],
            value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        ),
        const("axis", [0], []),
        helper.make_node("CumSum", ["ones", "axis"], ["pos"]),
        helper.make_node("Cast", ["pos"], ["pos_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["x", "pos_f"], ["y"]),
    ]
    graph = helper.make_graph(
        name="main_graph",
        nodes=nodes,
        inputs=[
            helper.make_value_info(
                "x", helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 8])
            ),
        ],
        outputs=[
            helper.make_value_info(
                "y", helper.make_tensor_type_proto(TensorProto.FLOAT, shape=[2, 8])
            ),
        ],
    )
    save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)]),
        "simplify_ones_cumsum_positions.onnx",
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
    sdpa_prescale_alias()
    constant_fold()
    expand_shape_chain()
    squeeze_shape_dim()
    reshape_concat_shape()
    resize_sizes_from_shape()
    pool_output_dims()
    reshape_symbolic_dims()
    pad_from_constants()
    ones_times_minus_one()
    expand_minus_one_sizes()
    ones_cumsum_positions()
    print("Done.")
