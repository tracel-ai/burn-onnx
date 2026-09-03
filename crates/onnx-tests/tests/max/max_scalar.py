#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/max/max_scalar.onnx

import numpy as np
import torch
import torch.nn as nn
from onnx.reference import ReferenceEvaluator


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, lhs_scalar, tensor, rhs_scalar):
        return (
            torch.maximum(lhs_scalar, tensor),
            torch.maximum(tensor, rhs_scalar),
            torch.maximum(lhs_scalar, rhs_scalar),
        )


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "max_scalar.onnx"

    lhs_scalar = torch.tensor(1.0)
    tensor = torch.tensor([[0.0, 1.5, -3.0, 2.5], [4.0, 1.0, -1.0, 3.5]], device=device)
    rhs_scalar = torch.tensor(2.0)
    torch.onnx.export(
        model,
        (lhs_scalar, tensor, rhs_scalar),
        onnx_name,
        verbose=False,
        input_names=["lhs_scalar", "tensor", "rhs_scalar"],
        output_names=["scalar_tensor", "tensor_scalar", "scalar_scalar"],
        opset_version=16,
        external_data=False,
    )

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input data: {} {} {}".format(lhs_scalar, tensor, rhs_scalar))
    output = model.forward(lhs_scalar, tensor, rhs_scalar)
    print("Test output data: {}".format(output))

    # Run the model using ReferenceEvaluator
    ref = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = ref.run(
        None,
        {
            "lhs_scalar": lhs_scalar.numpy(),
            "tensor": tensor.numpy(),
            "rhs_scalar": rhs_scalar.numpy(),
        },
    )

    max_scalar_tensor, max_tensor_scalar, max_scalar_scalar = outputs

    expected_lhs_scalar, expected_tensor, expected_rhs_scalar = model(
        lhs_scalar, tensor, rhs_scalar
    )
    np.testing.assert_allclose(max_scalar_tensor, expected_lhs_scalar.numpy())
    np.testing.assert_allclose(max_tensor_scalar, expected_tensor.numpy())
    np.testing.assert_allclose(max_scalar_scalar, expected_rhs_scalar.numpy())

    print(f"\nTest output max_scalar_tensor: {repr(max_scalar_tensor)}")
    print(f"Test output max_tensor_scalar: {repr(max_tensor_scalar)}")
    print(f"Test output max_scalar_scalar: {repr(max_scalar_scalar)}")


if __name__ == "__main__":
    main()
