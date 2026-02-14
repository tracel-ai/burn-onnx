#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: col2im.onnx and col2im_complex.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model_basic():
    # Col2Im: input [N, C * prod(block_shape), L], image_shape, block_shape -> output [N, C, *image_shape]
    # Match test_col2im_basic in mod.rs:
    # Image: [4, 4]
    # Block: [2, 2]
    # N=1, C=1
    # L = (4/2) * (4/2) = 4
    # Col Channels = 1 * 4 = 4
    # Input: [1, 4, 4]
    
    input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 4])
    output_tensor = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 4, 4]
    )

    # Create the Col2Im node
    col2im_node = onnx.helper.make_node(
        "Col2Im",
        inputs=["input", "image_shape", "block_shape"],
        outputs=["output"],
        name="Col2ImNode",
        strides=[2, 2],
    )

    # Create constant initializers for image_shape and block_shape
    image_shape_init = onnx.numpy_helper.from_array(
        np.array([4, 4], dtype=np.int64), name="image_shape"
    )
    block_shape_init = onnx.numpy_helper.from_array(
        np.array([2, 2], dtype=np.int64), name="block_shape"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [col2im_node],
        "Col2ImModel",
        [input_tensor],
        [output_tensor],
        initializer=[image_shape_init, block_shape_init],
    )

    # Create the model (opset 18 required for Col2Im)
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=graph,
        producer_name="ONNX_Generator",
    )

    return model


def build_model_complex():
    # Complex case:
    # Image Shape: [5, 5]
    # Block Shape: [2, 2]
    # Strides: [2, 2]
    # Pads: [1, 1, 1, 1] (1 pad on all sides)
    # Dilations: [2, 2]
    
    # N=1, C=1
    # Effective Block: dilation * (kernel-1) + 1 = 2 * (2-1) + 1 = 3
    # Output H = (5 + 1 + 1 - 3) / 2 + 1 = 4 / 2 + 1 = 3
    # Output W = (5 + 1 + 1 - 3) / 2 + 1 = 3
    # L = 3 * 3 = 9
    # Col Channels = 1 * (2*2) = 4
    # Input Shape: [1, 4, 9] => [1, 36] flattened? No, [1, 4, 9] (N, C*BlockProd, L)
    
    # Wait, the integration test in mod.rs says:
    # "Input: [1, 9, 9] from col2im.py (N=1, C_in=9, L=9)"
    # "Expected Output Shape: [1, 1, 5, 5]"
    # "(Image 5x5, N=1, C_out=1)"
    
    # Let's match the values in mod.rs comments:
    # C_in = 9, L = 9. BlockProd must be 9 (since C_out=1 usually).
    # Block Shape likely [3, 3] then.
    # Image [5, 5]
    
    # Re-reading mod.rs comment:
    # "We use ones [1, 9, 9]... Expected Output [1, 1, 5, 5]"
    
    # Let's define parameters that result in this:
    # Image: [5, 5]
    # Block: [3, 3] -> BlockProd 9.
    # Input Channels: 1 * 9 = 9.
    # L = 9.
    # Output spatial (blocks): 3x3.
    # We need L = H_out * W_out = 9. So 3x3 grid of blocks.
    
    # Calculation:
    # (H + pad_h - effective_kernel) / stride + 1 = 3
    # (5 + pad_h - effective_kernel) / stride + 1 = 3
    
    # Let's try:
    # Block [3, 3], Dilation [1, 1] => Effective [3, 3]
    # Stride [2, 2]?
    # (5 + pad - 3) / 2 + 1 = 3
    # (2 + pad) / 2 + 1 = 3
    # (2 + pad) / 2 = 2
    # 2 + pad = 4 => pad = 2 (total). So pads=[1, 1, 1, 1].
    
    # So: Image=[5,5], Block=[3,3], Strides=[2,2], Pads=[1,1,1,1], Dilations=[1,1].
    
    input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 9, 9])
    output_tensor = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 5, 5]
    )

    # Create the Col2Im node
    col2im_node = onnx.helper.make_node(
        "Col2Im",
        inputs=["input", "image_shape", "block_shape"],
        outputs=["output"],
        name="Col2ImNode",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1], # Explicitly set default
    )

    # Create constant initializers
    image_shape_init = onnx.numpy_helper.from_array(
        np.array([5, 5], dtype=np.int64), name="image_shape"
    )
    block_shape_init = onnx.numpy_helper.from_array(
        np.array([3, 3], dtype=np.int64), name="block_shape"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [col2im_node],
        "Col2ImModel",
        [input_tensor],
        [output_tensor],
        initializer=[image_shape_init, block_shape_init],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=graph,
        producer_name="ONNX_Generator",
    )

    return model

def save_model(model, filename):
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Finished exporting model to {filename}")


def main():
    print("Building Col2Im models...")

    # Basic
    model_basic = build_model_basic()
    save_model(model_basic, "col2im_basic.onnx")

    # Complex
    model_complex = build_model_complex()
    save_model(model_complex, "col2im_complex.onnx")

    # Run reference on complex
    # Use pattern inputs to verify index mapping, not just shape/sum
    test_input = np.arange(1, 82, dtype=np.float32).reshape(1, 9, 9)
    session = onnx.reference.ReferenceEvaluator(model_complex)
    (test_output,) = session.run(None, {"input": test_input})
    print(f"Complex model output shape: {test_output.shape}")
    print("Complex model reference output (flattened):")
    print(list(test_output.flatten()))


if __name__ == "__main__":
    main()
