#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scaler.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 1


def main():
    # Test case: Apply scaling with both scale and offset
    np.random.seed(42)
    
    # Create input data
    input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    # Define Scaler node - applies formula: Y = (X - offset) * scale
    # With scale=2.0 and offset=1.0: Y = (X - 1.0) * 2.0
    node = helper.make_node(
        "Scaler",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        scale=[2.0],
        offset=[1.0],
    )
    
    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    
    graph = helper.make_graph(
        [node],
        "scaler_test",
        [input_tensor],
        [output_tensor],
    )
    
    model = helper.make_model(
        graph, opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),  # Default domain opset
        ]
    )
    
    onnx.save(model, "scaler.onnx")
    print(f"Finished exporting model to scaler.onnx")
    
    # Validate using ReferenceEvaluator
    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})
    
    print("\nInput:")
    print(input_data)
    print("\nOutput (Y = (X - 1.0) * 2.0):")
    print(result[0])
    print("\nExpected:")
    print((input_data - 1.0) * 2.0)
    
    # Save test data
    np.save("input.npy", input_data)
    np.save("output.npy", result[0])


if __name__ == "__main__":
    main()
