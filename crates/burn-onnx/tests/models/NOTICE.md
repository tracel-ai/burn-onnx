# ResNet integration-test fixture

The ResNet-18 inference fixture in `resnet.rs` is adapted from the `resnet-burn` model in the
[models repository](https://github.com/tracel-ai/models).

The fixture intentionally excludes pretrained weights, download code, training code, and model
variants that are not exercised by the ONNX exporter test.
