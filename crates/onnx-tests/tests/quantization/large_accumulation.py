# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
w=np.full((1,512,1),255,np.uint8); w[0,0,0]=254
x=np.full((1,512,1),255,np.uint8)
m=h.make_model(h.make_graph([h.make_node('ConvInteger',['x','w'],['y'])],'large',[h.make_tensor_value_info('x',T.UINT8,[1,512,1])],[h.make_tensor_value_info('y',T.INT32,[1,1,1])],[nh.from_array(w,'w')]),opset_imports=[h.make_opsetid('',10)])
onnx.checker.check_model(m)
print(ReferenceEvaluator(m).run(None,{'x':x}))
onnx.save(m,Path(__file__).with_suffix('.onnx'))
