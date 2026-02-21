use super::prelude::*;

impl NodeCodegen for onnx_ir::shrink::ShrinkNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let lambd = self.config.lambd.to_tokens();
        let bias = self.config.bias.to_tokens();

        quote! {
        let #output = burn::tensor::activation::shrink(#input, #lambd, #bias);
        }
    }
}
