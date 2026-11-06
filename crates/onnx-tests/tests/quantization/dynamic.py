# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
info=lambda n,t,s:h.make_tensor_value_info(n,t,s)
m=h.make_model(h.make_graph([h.make_node('DynamicQuantizeLinear',['x'],['y','scale','zero'])],'dynamic',[info('x',T.FLOAT,[2,3])],[info('y',T.UINT8,[2,3]),info('scale',T.FLOAT,[]),info('zero',T.UINT8,[])]),opset_imports=[h.make_opsetid('',11)])
onnx.checker.check_model(m)
onnx.save(m,Path(__file__).with_suffix('.onnx'))
for x in [np.array([[-2,-1,0],[1,2,3]],np.float32),np.zeros((2,3),np.float32),np.array([[1,2,3],[4,5,6]],np.float32),np.array([[-6,-5,-4],[-3,-2,-1]],np.float32)]:
 print(ReferenceEvaluator(m).run(None,{'x':x}))
