use super::prelude::*;

impl NodeCodegen for onnx_ir::node::max_pool3d::MaxPool3dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        panic!(
            "burn-onnx does not currently support 3D max pooling: Burn lacks a native MaxPool3d \
             primitive. Tracking issue: https://github.com/tracel-ai/burn-onnx/issues/343"
        );
    }

    fn forward(&self, _scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        panic!(
            "burn-onnx does not currently support 3D max pooling: Burn lacks a native MaxPool3d \
             primitive. Tracking issue: https://github.com/tracel-ai/burn-onnx/issues/343"
        );
    }

    fn register_imports(&self, _imports: &mut BurnImports) {}
}
