#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
#   "ml_dtypes",
# ]
# ///

# Generates qlinear_matmul.onnx and sanity-checks with ReferenceEvaluator.

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

_NP_TO_PROTO = {
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
}


def make_value_info(name, np_val, proto_type):
    shape = [] if np_val.ndim == 0 else list(np_val.shape)
    return helper.make_tensor_value_info(name, proto_type, shape)


def build_model(
    suffix: str,
    a: np.ndarray,
    a_scale: np.ndarray,
    a_zero_point: np.ndarray,
    b: np.ndarray,
    b_scale: np.ndarray,
    b_zero_point: np.ndarray,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    opset: int = 21,
) -> None:
    np.random.seed(42)

    print(f"\n--- QLinearMatMul ({suffix}) ---")

    a_proto = _NP_TO_PROTO[a.dtype.name]
    b_proto = _NP_TO_PROTO[b.dtype.name]
    y_zero_point_proto = _NP_TO_PROTO[y_zero_point.dtype.name]
    a_scale_proto = _NP_TO_PROTO[a_scale.dtype.name]
    b_scale_proto = _NP_TO_PROTO[b_scale.dtype.name]
    y_scale_proto = _NP_TO_PROTO[y_scale.dtype.name]
    y_shape = list(a.shape[:-1]) + [b.shape[-1]]

    node = helper.make_node(
        "QLinearMatMul",
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        name="qlinear_matmul",
    )

    graph = helper.make_graph(
        [node],
        f"qlinear_matmul_{suffix}_graph",
        [
            make_value_info("a", a, a_proto),
            make_value_info("a_scale", a_scale, a_scale_proto),
            make_value_info("a_zero_point", a_zero_point, a_proto),
            make_value_info("b", b, b_proto),
            make_value_info("b_scale", b_scale, b_scale_proto),
            make_value_info("b_zero_point", b_zero_point, b_proto),
            make_value_info("y_scale", y_scale, y_scale_proto),
            make_value_info("y_zero_point", y_zero_point, y_zero_point_proto),
        ],
        [helper.make_tensor_value_info("y", y_zero_point_proto, y_shape)],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    model_path = f"qlinear_matmul_{suffix}.onnx"
    onnx.save(model, model_path)
    print(f"ONNX model saved to {model_path}")

    # The ReferenceEvaluator requires scale/zero_point tensors to be broadcastable
    # with their operand. Rank-1 vectors (as allowed by the spec) must be unsqueezed
    # to the correct axis before evaluation: (M,) -> (M, 1) for per-row, (N,) -> (1, N)
    # for per-column.
    def _expand_dim(scale, axis):
        if scale.ndim == 1:
            return np.expand_dims(scale, axis=axis)
        return scale

    input_dict = {
        "a": a,
        "a_scale": _expand_dim(a_scale, axis=-1),
        "a_zero_point": _expand_dim(a_zero_point, axis=-1),
        "b": b,
        "b_scale": _expand_dim(b_scale, axis=0),
        "b_zero_point": _expand_dim(b_zero_point, axis=0),
        "y_scale": _expand_dim(y_scale, axis=-1),
        "y_zero_point": _expand_dim(y_zero_point, axis=-1),
    }
    output = ReferenceEvaluator(model).run(None, input_dict)[0]

    for k, v in input_dict.items():
        print(
            f"{k} shape: {v.shape if isinstance(v, np.ndarray) else 'scalar'}\n{k}: {v}\n"
        )
    print(f"Output y shape: {output.shape}\nOutput y: {output}\n")


if __name__ == "__main__":
    a_shape_3d = [2, 2, 4]
    b_shape_3d = [2, 4, 3]
    a_shape_2d = [2, 4]
    b_shape_2d = [4, 3]

    # Case 1: Scalar scales/zero-points (per-tensor quantization)
    # Input range limited to [0, 20) to prevent saturation:
    #   max dot product ≈ 4×17×16=1088, max output ≈ 1088×0.02/0.3+4≈76
    build_model(
        "scalar",
        a=np.random.randint(0, 20, size=a_shape_3d, dtype=np.uint8),
        a_scale=np.array(0.1, dtype=np.float32),
        a_zero_point=np.array(2, dtype=np.uint8),
        b=np.random.randint(0, 20, size=b_shape_3d, dtype=np.uint8),
        b_scale=np.array(0.2, dtype=np.float32),
        b_zero_point=np.array(3, dtype=np.uint8),
        y_scale=np.array(0.3, dtype=np.float32),
        y_zero_point=np.array(4, dtype=np.uint8),
    )

    # Case 2: Vector scales/zero-points (2D operands).
    # `a` uses a row vector (one scale per row of a, shape [M]) — quantizes along axis 0.
    # `b` uses a column vector (one scale per column of b, shape [N]) — quantizes along axis 1.
    # `y` uses a row vector (one scale per row of output, shape [M]).
    # Input range [0, 20): max output ≈ 4×19×19×1.0/0.5+5≈1455, but limited by per-row scaling
    M, _ = a_shape_2d
    _, N = b_shape_2d
    build_model(
        "vector",
        a=np.random.randint(0, 20, size=a_shape_2d, dtype=np.uint8),
        a_scale=np.random.uniform(0.01, 1.0, size=(M,)).astype(
            np.float32
        ),  # row vector [M]
        a_zero_point=np.random.randint(0, 5, size=(M,)).astype(
            np.uint8
        ),  # row vector [M]
        b=np.random.randint(0, 20, size=b_shape_2d, dtype=np.uint8),
        b_scale=np.random.uniform(0.01, 1.0, size=(N,)).astype(
            np.float32
        ),  # column vector [N]
        b_zero_point=np.random.randint(0, 5, size=(N,)).astype(
            np.uint8
        ),  # column vector [N]
        y_scale=np.array([1.0, 0.5], dtype=np.float32),  # row vector [M]
        y_zero_point=np.array([10, 5], dtype=np.uint8),  # row vector [M]
    )

    # Case 3: N-D scales/zero-points (3D operands, per-row for `a`, per-column for `b`, per-row for `y`).
    batch, M, _ = a_shape_3d
    _, _, N = b_shape_3d
    build_model(
        "nd",
        a=np.random.randint(0, 20, size=a_shape_3d, dtype=np.uint8),
        a_scale=np.random.uniform(0.01, 1.0, size=(batch, M, 1)).astype(np.float32),
        a_zero_point=np.random.randint(0, 5, size=(batch, M, 1)).astype(np.uint8),
        b=np.random.randint(0, 20, size=b_shape_3d, dtype=np.uint8),
        b_scale=np.random.uniform(0.01, 1.0, size=(batch, 1, N)).astype(np.float32),
        b_zero_point=np.random.randint(0, 5, size=(batch, 1, N)).astype(np.uint8),
        y_scale=np.random.uniform(0.5, 2.0, size=(batch, M, 1)).astype(np.float32),
        y_zero_point=np.random.randint(0, 15, size=(batch, M, 1)).astype(np.uint8),
    )

    # Case 4: U8 saturation test — engineered to hit both the upper (255) and lower (0) U8 clamp.
    # Uses large values (120) to overflow upward and zero values to overflow downward (via negative
    # a_int after zero_point subtraction). The ONNX ReferenceEvaluator wraps on overflow rather
    # than saturating, so expected values are hand-computed using explicit np.clip():
    #   output = clip(round(a_scale * b_scale * matmul(a-a_zp, b-b_zp) / y_scale) + y_zp, 0, 255)
    #   [0,0]: round(4×118×117×0.1x0.2/0.3)+4 = 3686 → 255  (upper saturation)
    #   [0,1]: round(4×118×2×0.1x0.2/0.3)+4   =   67        (no saturation)
    #   [1,0]: round(4×-2×117×0.1x0.2/0.3)+4  =  -58 →   0  (lower saturation)
    #   [1,1]: round(4×-2×2×0.1x0.2/0.3)+4    =    3        (no saturation)
    build_model(
        "u8_saturate",
        a=np.array([[120, 120, 120, 120], [0, 0, 0, 0]], dtype=np.uint8),
        a_scale=np.array(0.1, dtype=np.float32),
        a_zero_point=np.array(2, dtype=np.uint8),
        b=np.array([[120, 5], [120, 5], [120, 5], [120, 5]], dtype=np.uint8),
        b_scale=np.array(0.2, dtype=np.float32),
        b_zero_point=np.array(3, dtype=np.uint8),
        y_scale=np.array(0.3, dtype=np.float32),
        y_zero_point=np.array(4, dtype=np.uint8),
    )

    # Case 5: I8 saturation test — exercises clamp(-128, 127) and the I8 cast.
    # Uses symmetric quantization (zero_point=0) with large positive and large negative values.
    # The ONNX ReferenceEvaluator wraps on overflow rather than saturating, so expected values
    # are hand-computed using explicit np.clip():
    #   output = clip(round(a_scale * b_scale * matmul(a-a_zp, b-b_zp) / y_scale) + y_zp, -128, 127)
    #   [0,0]: round(4×100×100×0.1×0.2/0.1)+0 = 8000 → 127   (upper saturation)
    #   [0,1]: round(4×100×1×0.1×0.2/0.1)+0   =   80          (no saturation)
    #   [1,0]: round(4×-100×100×0.1×0.2/0.1)+0 = -8000 → -128 (lower saturation)
    #   [1,1]: round(4×-100×1×0.1×0.2/0.1)+0   =  -80          (no saturation)
    build_model(
        "i8_saturate",
        a=np.array([[100, 100, 100, 100], [-100, -100, -100, -100]], dtype=np.int8),
        a_scale=np.array(0.1, dtype=np.float32),
        a_zero_point=np.array(0, dtype=np.int8),
        b=np.array([[100, 1], [100, 1], [100, 1], [100, 1]], dtype=np.int8),
        b_scale=np.array(0.2, dtype=np.float32),
        b_zero_point=np.array(0, dtype=np.int8),
        y_scale=np.array(0.1, dtype=np.float32),
        y_zero_point=np.array(0, dtype=np.int8),
    )

    # Case 6: opset-10 model (2D operands, U8 scalar, F32 scales)
    # Opset-10 only supports FLOAT scales; the test verifies legacy model loading.
    build_model(
        "opset_10",
        a=np.random.randint(0, 20, size=[2, 4], dtype=np.uint8),
        a_scale=np.array(0.1, dtype=np.float32),
        a_zero_point=np.array(2, dtype=np.uint8),
        b=np.random.randint(0, 20, size=[4, 3], dtype=np.uint8),
        b_scale=np.array(0.2, dtype=np.float32),
        b_zero_point=np.array(3, dtype=np.uint8),
        y_scale=np.array(0.3, dtype=np.float32),
        y_zero_point=np.array(4, dtype=np.uint8),
        opset=10,
    )

    # Case 7: scalar F16 scales — exercises the `(scale as f32)` cast path in generated code.
    # Uses the same operand values as Case 6 for easy cross-reference.
    build_model(
        "scalar_f16_scale",
        a=np.array([[6, 1, 19, 10], [11, 3, 2, 19]], dtype=np.uint8),
        a_scale=np.array(0.1, dtype=np.float16),
        a_zero_point=np.array(2, dtype=np.uint8),
        b=np.array([[14, 14, 10], [3, 7, 7], [14, 1, 12], [11, 6, 16]], dtype=np.uint8),
        b_scale=np.array(0.2, dtype=np.float16),
        b_zero_point=np.array(3, dtype=np.uint8),
        y_scale=np.array(0.3, dtype=np.float16),
        y_zero_point=np.array(4, dtype=np.uint8),
    )

    # Case 8: vector BF16 scales — exercises the `.cast(DType::F32)` path for rank-1 scale tensors.
    # Uses the same operand values as Case 2 for easy cross-reference.
    # NOTE: The Rust test expects [88, 254, 22] at row 1, not [87, ...] from ReferenceEvaluator.
    # The ReferenceEvaluator computes in BF16 throughout; Rust casts BF16→F32 then computes in
    # F32. For [1,0]: BF16 intermediate product rounds to 0.11816 (→ 87), F32 gives 0.11825 (→ 88).
    build_model(
        "vector_bf16_scale",
        a=np.array([[6, 1, 19, 10], [11, 3, 2, 19]], dtype=np.uint8),
        a_scale=np.array([0.19160044, 0.7818941], dtype=ml_dtypes.bfloat16),
        a_zero_point=np.array([4, 1], dtype=np.uint8),
        b=np.array([[18, 1, 15], [7, 10, 17], [14, 10, 17], [13, 13, 3]], dtype=np.uint8),
        b_scale=np.array([0.15143815, 0.6543796, 0.06584746], dtype=ml_dtypes.bfloat16),
        b_zero_point=np.array([3, 1, 3], dtype=np.uint8),
        y_scale=np.array([1.0, 0.5], dtype=ml_dtypes.bfloat16),
        y_zero_point=np.array([10, 5], dtype=np.uint8),
    )
