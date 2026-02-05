#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "onnx==1.19.0",
# ]
# ///

# used to generate model: onnx-tests/tests/sign/sign.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.sign(x)
        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "sign.onnx"
    test_input = torch.tensor([[[[-1.0, 2.0, 0.0, -4.0]]]])

    torch.onnx.export(
        model,
        test_input,
        onnx_name,
        verbose=False,
        opset_version=16,
        external_data=False,
    )
    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
