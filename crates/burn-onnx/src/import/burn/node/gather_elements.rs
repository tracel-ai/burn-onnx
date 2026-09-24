use super::prelude::*;

impl NodeCodegen for onnx_ir::gather_elements::GatherElementsNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> proc_macro2::TokenStream {
        let dim = self.config.axis.to_tokens();
        let input = scope.arg(self.inputs.first().unwrap());
        let index = scope.arg(&self.inputs[1]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        // ONNX allows indices down to `-dim_size` along the gather axis, which burn's
        // `gather` does not accept, so fold negatives first. Indices outside
        // `[-dim_size, dim_size - 1]` are an error per the ONNX spec and stay unchecked,
        // like ScatterElements.
        quote! {
            let #output = {
                let axis_size = #input.dims()[#dim] as i64;
                let negative = #index.clone().lower_elem(0i64);
                let corrected = #index.clone() + axis_size;
                let indices = #index.mask_where(negative, corrected);
                #input.gather(#dim, indices)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gather_elements::{
        GatherElementsConfig, GatherElementsInput, GatherElementsNodeBuilder,
    };

    #[test]
    fn test_gather_elements() {
        let config = GatherElementsConfig {
            indices: GatherElementsInput::Static(vec![]),
            axis: 1,
        };
        let node = GatherElementsNodeBuilder::new("gather1")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, indices: Tensor<2, Int>) -> Tensor<2> {
            let output = {
                let axis_size = input.dims()[1] as i64;
                let negative = indices.clone().lower_elem(0i64);
                let corrected = indices.clone() + axis_size;
                let indices = indices.mask_where(negative, corrected);
                input.gather(1, indices)
            };
            output
        }
        ");
    }
}
