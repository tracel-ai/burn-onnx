# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["onnx==1.19.0", "numpy<2"]
# ///
"""Copy the official expanded functions and verify their reference data."""
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
root = Path(__file__).resolve().parent
vendor = root.parents[2] / "onnx-official-tests" / "vendor" / "node"
for dimension in ("2d", "3d"):
    for alignment in ("", "align_corners_"):
        name = f"affine_grid_{dimension}_{alignment}expanded"
        source = vendor / f"test_{name}"
        model = onnx.load(source / "model.onnx")
        feeds = {}
        for i, info in enumerate(model.graph.input):
            proto = onnx.TensorProto()
            proto.ParseFromString((source / "test_data_set_0" / f"input_{i}.pb").read_bytes())
            feeds[info.name] = numpy_helper.to_array(proto)
        expected = ReferenceEvaluator(model).run(None, feeds)[0]
        onnx.save(model, root / f"{name}.onnx")
        feeds["theta"].astype("<f4").tofile(root / f"{name}_theta.bin")
        expected.astype("<f4").tofile(root / f"{name}_expected.bin")
        feeds["size"] = np.array([2, 3, 2, 3] if dimension == "2d" else [2, 3, 3, 2, 3], dtype=np.int64)
        ReferenceEvaluator(model).run(None, feeds)[0].astype("<f4").tofile(root / f"{name}_small_expected.bin")
        print(name, feeds["theta"].shape, feeds["size"].tolist(), expected.shape)
