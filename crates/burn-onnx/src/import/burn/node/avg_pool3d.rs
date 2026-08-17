use super::prelude::*;

impl NodeCodegen for onnx_ir::node::avg_pool3d::AveragePool3dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        panic!(
            "burn-onnx does not currently support 3D average pooling: Burn lacks a native AvgPool3d \
             primitive. Tracking issue: https://github.com/tracel-ai/burn-onnx/issues/343"
        );
    }

    fn forward(&self, _scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        panic!(
            "burn-onnx does not currently support 3D average pooling: Burn lacks a native AvgPool3d \
             primitive. Tracking issue: https://github.com/tracel-ai/burn-onnx/issues/343"
        );
    }

    fn register_imports(&self, _imports: &mut BurnImports) {}
}
