use super::prelude::*;

impl NodeCodegen for onnx_ir::node::multinomial::MultinomialNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        if let Some(seed) = self.config.seed {
            log::warn!(
                "Node '{}' (Multinomial): seed attribute ({seed}) is ignored; samples use the backend random generator",
                self.name
            );
        }
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);
        let sample_size = self.config.sample_size;
        let dtype = self.config.dtype.to_tokens();

        // The input holds unnormalized log-probabilities. Subtracting the row max keeps `exp`
        // from overflowing; `categorical` normalizes the weights, so the shift does not change
        // the distribution.
        quote! {
            let #output = {
                let logits = #input;
                let max = logits.clone().max_dim(1);
                (logits - max).exp().categorical(#sample_size).cast(#dtype)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::multinomial::{MultinomialConfig, MultinomialNodeBuilder};

    #[test]
    fn test_multinomial_int32_default() {
        let node = MultinomialNodeBuilder::new("multinomial1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::I32)
            .config(MultinomialConfig::default())
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2, Int> {
            let output = {
                let logits = input;
                let max = logits.clone().max_dim(1);
                (logits - max).exp().categorical(1usize).cast(burn::tensor::DType::I32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_multinomial_int64_samples() {
        let config = MultinomialConfig {
            dtype: DType::I64,
            sample_size: 10,
            seed: Some(3.0),
        };
        let node = MultinomialNodeBuilder::new("multinomial2")
            .input_tensor("input", 2, DType::F64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2, Int> {
            let output = {
                let logits = input;
                let max = logits.clone().max_dim(1);
                (logits - max).exp().categorical(10usize).cast(burn::tensor::DType::I64)
            };
            output
        }
        ");
    }
    #[test]
    fn temporary_names_can_be_graph_inputs() {
        for name in ["logits", "max"] {
            let node = MultinomialNodeBuilder::new("sample")
                .input_tensor(name, 2, DType::F32)
                .output_tensor("output", 2, DType::I32)
                .config(MultinomialConfig::default())
                .build();
            shadow_check_result(&node).unwrap();
        }
    }
}
