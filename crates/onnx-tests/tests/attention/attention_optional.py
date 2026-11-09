# /// script
# dependencies = ["onnx==1.23.0", "numpy"]
# ///
"""Empty past/present slots must remain absent, even before a qk output."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
root = Path(__file__).parent
shape = [1, 1, 2, 2]
feeds = {"q": np.array([1,0,0,1], dtype=np.float32).reshape(shape),
         "k": np.array([0,1,1,0], dtype=np.float32).reshape(shape),
         "v": np.array([.25,.5,.3,.6], dtype=np.float32).reshape(shape)}
for suffix, outputs in [("native", ["y", "", "", ""]), ("qk", ["y", "", "", "qk"])]:
    names = [n for n in outputs if n]
    model = h.make_model(h.make_graph(
        [h.make_node("Attention", ["q", "k", "v", "", "", ""], outputs, qk_matmul_output_mode=0)],
        "optional", [h.make_tensor_value_info(n, T.FLOAT, shape) for n in feeds],
        [h.make_tensor_value_info(n, T.FLOAT, shape) for n in names]),
        opset_imports=[h.make_opsetid("", 23)])
    onnx.checker.check_model(model)
    onnx.save(model, root / f"attention_optional_{suffix}.onnx")
    for name, output in zip(names, ReferenceEvaluator(model).run(None, feeds)):
        output.astype("<f4").tofile(root / f"attention_optional_{suffix}_{name}.bin")
