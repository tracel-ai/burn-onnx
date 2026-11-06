# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
m = h.make_model(h.make_graph([h.make_node("Expand", ["x", "shape"], ["y"])], "partial_expand", [h.make_tensor_value_info("x", T.FLOAT, ["N",3]), h.make_tensor_value_info("shape", T.INT64, [2])], [h.make_tensor_value_info("y", T.FLOAT, ["M",3])]), opset_imports=[h.make_opsetid("",18)])
onnx.checker.check_model(m)
x = np.array([[1,2,3]], dtype=np.float32)
np.testing.assert_array_equal(ReferenceEvaluator(m).run(None, {"x":x,"shape":np.array([4,3], dtype=np.int64)})[0], np.tile(x,(4,1)))
onnx.save(m, Path(__file__).with_suffix(".onnx"))
