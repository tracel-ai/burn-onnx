# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Shape contracts refer to original input names, even after collision renaming."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
for name, first, second in [
    ("shape_input_collision", "a-b", "a_b"),
    ("shape_input_renamed", "a.b", "a-b"),
]:
    model = h.make_model(h.make_graph(
        [h.make_node("Size", [first], ["first_size"]),
         h.make_node("Size", [second], ["shape_size"])], name,
        [h.make_tensor_value_info(first, T.INT64, [2]),
         h.make_tensor_value_info(second, T.INT64, [4])],
        [h.make_tensor_value_info("first_size", T.INT64, []),
         h.make_tensor_value_info("shape_size", T.INT64, [])],
    ), opset_imports=[h.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    result = ReferenceEvaluator(model).run(None, {
        first: np.array([7, 8, 9], np.int64),
        second: np.array([10, 20, 30, 40], np.int64),
    })
    assert [int(value) for value in result] == [3, 4]
    onnx.save(model, Path(__file__).with_name(name + ".onnx"))
