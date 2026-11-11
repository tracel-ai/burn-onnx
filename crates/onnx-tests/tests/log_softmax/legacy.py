#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnx==1.23.2", "numpy"]
# ///
"""Generate old-opset normalization fixtures and modern controls."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

OUTPUT = Path(__file__).resolve().parent
OPERATOR = "LogSoftmax"
for suffix,opset,shape,axis in [
    ("legacy_axis0",11,[2,2],0),
    ("legacy_default",11,[1,2,2],None),
    ("legacy_negative",12,[1,2,2],-2),
    ("modern_axis0",13,[2,2],0),
]:
    attributes = {} if axis is None else {"axis":axis}
    model = helper.make_model(helper.make_graph(
        [helper.make_node(OPERATOR,["input"],["output"],**attributes)],
        suffix,[helper.make_tensor_value_info("input",TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info("output",TensorProto.FLOAT,shape)]),
        opset_imports=[helper.make_opsetid("",opset)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,OUTPUT/("log_softmax_"+suffix+".onnx"))
    values=np.arange(4,dtype=np.float32).reshape(shape)
    # The reference evaluator's legacy normalization does not coerce to 2D.
    # Evaluate the equivalent rank-2 graph explicitly for pre-13 fixtures.
    reference_model=helper.make_model(helper.make_graph(
        [helper.make_node(OPERATOR,["input"],["output"],axis=1 if opset<13 else axis)],
        "reference",[helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,4] if opset<13 else shape)],
        [helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,4] if opset<13 else shape)]),
        opset_imports=[helper.make_opsetid("",13)],ir_version=10)
    expected=ReferenceEvaluator(reference_model).run(None,{"input":values.reshape(1,4) if opset<13 else values})[0].reshape(shape)
    print(suffix,expected)
