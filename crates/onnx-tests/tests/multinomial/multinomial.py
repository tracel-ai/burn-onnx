#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# Used to generate models: multinomial.onnx, multinomial_int64.onnx
#
# Multinomial draws random samples, so there is no reference output to compare against.
# The Rust test checks shape, dtype, index range, and that a one-hot input (one class at
# log-probability 0, the others at -inf or a very negative value) always samples that class.
# onnx.reference.ReferenceEvaluator has no Multinomial implementation, so the models are only
# validated with onnx.checker here.

from pathlib import Path
import onnx
import onnx.helper
from onnx import TensorProto


def build_model(name, sample_size, dtype=None, seed=None, shape=(3,4), input_dtype=TensorProto.FLOAT):
    attrs = {} if sample_size == 1 else {"sample_size": sample_size}
    if dtype is not None:
        attrs["dtype"] = dtype
    if seed is not None:
        attrs["seed"] = seed

    node = onnx.helper.make_node(
        "Multinomial",
        inputs=["input"],
        outputs=["output"],
        name="/Multinomial",
        **attrs,
    )

    out_type = TensorProto.INT32 if dtype in (None, TensorProto.INT32) else dtype
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[node],
        inputs=[onnx.helper.make_tensor_value_info("input", input_dtype, list(shape))],
        outputs=[
            onnx.helper.make_tensor_value_info("output", out_type, [shape[0], sample_size])
        ],
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", 22)],
        producer_name="ONNX_Generator",
    )
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_name(name))
    print(f"Finished exporting model to {name}")


def main():
    build_model("multinomial.onnx", sample_size=16)
    build_model("multinomial_default.onnx", sample_size=1)
    build_model(
        "multinomial_int64.onnx", sample_size=8, dtype=TensorProto.INT64, seed=7.0
    )


if __name__ == "__main__":
    main()

build_model("multinomial_half.onnx", 32, shape=(1,65536), input_dtype=TensorProto.FLOAT16)
build_model("multinomial_empty.onnx", 2, shape=(0,4))
model = onnx.helper.make_model(onnx.helper.make_graph([
    onnx.helper.make_node("Multinomial", ["input"], ["a"], sample_size=128),
    onnx.helper.make_node("Multinomial", ["input"], ["b"], sample_size=128),
    onnx.helper.make_node("Add", ["a", "b"], ["sum"]),
], "independent_draws", [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,2])], [onnx.helper.make_tensor_value_info("sum", TensorProto.INT32, [1,128])]), opset_imports=[onnx.helper.make_opsetid("", 22)])
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).parent.parent / "simplify" / "simplify_multinomial.onnx")
