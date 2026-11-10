use super::prelude::*;
use onnx_ir::ir::DType;

impl NodeCodegen for onnx_ir::node::multinomial::MultinomialNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let input = if matches!(self.inputs[0].ty.elem_type(), DType::F16 | DType::BF16) {
            quote! { #input.cast(burn::tensor::DType::F32) }
        } else {
            input
        };
        let output = arg_to_ident(&self.outputs[0]);
        let sample_size = self.config.sample_size;
        let dtype = self.config.dtype.to_tokens();

        let sample = if self.config.seed.is_some() {
            let draw = super::random_helpers::random_draw(
                self.config.seed,
                quote! { [weights.dims()[0], #sample_size] },
                quote! { burn::tensor::Distribution::Uniform(0.0, 1.0) },
                quote! { &weights.device() },
                quote! { weights.dtype() },
            );
            // Same tensor inverse-CDF method as Burn's categorical(), with an explicit key.
            quote! {
                let categories = weights.dims()[1];
                let sum = weights.clone().sum_dim(1);
                let uniform: Tensor<2> = #draw;
                let cdf = (weights / sum).cumsum(1).unsqueeze_dim::<3>(2);
                cdf.lower(uniform.unsqueeze_dim::<3>(1)).int()
                    .cast(burn::tensor::DType::I64).sum_dim(1).squeeze_dim::<2>(1)
                    .clamp(0, categories as i64 - 1).cast(#dtype)
            }
        } else {
            quote! { weights.categorical(#sample_size).cast(#dtype) }
        };

        // The input holds unnormalized log-probabilities. Subtracting the row max keeps `exp`
        // from overflowing; `categorical` normalizes the weights, so the shift does not change
        // the distribution.
        quote! {
            let #output = {
                let logits = #input;
                if logits.dims()[0] == 0 {
                    burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::new(alloc::vec::Vec::<i64>::new(), [0, #sample_size]),
                        (&logits.device(), #dtype),
                    )
                } else {
                    let max = logits.clone().max_dim(1);
                    let weights = (logits - max).exp();
                    #sample
                }
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
                if logits.dims()[0] == 0 {
                    burn::tensor::Tensor::<
                        2,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::new(
                            alloc::vec::Vec::<i64>::new(),
                            [0, 1usize],
                        ),
                        (&logits.device(), burn::tensor::DType::I32),
                    )
                } else {
                    let max = logits.clone().max_dim(1);
                    let weights = (logits - max).exp();
                    weights.categorical(1usize).cast(burn::tensor::DType::I32)
                }
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
                if logits.dims()[0] == 0 {
                    burn::tensor::Tensor::<
                        2,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::new(
                            alloc::vec::Vec::<i64>::new(),
                            [0, 10usize],
                        ),
                        (&logits.device(), burn::tensor::DType::I64),
                    )
                } else {
                    let max = logits.clone().max_dim(1);
                    let weights = (logits - max).exp();
                    let categories = weights.dims()[1];
                    let sum = weights.clone().sum_dim(1);
                    let uniform: Tensor<2> = Tensor::from_dispatch(
                        <burn::backend::Dispatch as burn::backend::ops::FloatTensorOps<
                            burn::backend::Dispatch,
                        >>::float_random(
                            ([weights.dims()[0], 10usize]).into(),
                            burn::tensor::Distribution::Uniform(0.0, 1.0),
                            (&weights.device()).as_dispatch(),
                            (weights.dtype()).into(),
                            burn::backend::RandomKey::new(1077936128u64),
                        ),
                    );
                    let cdf = (weights / sum).cumsum(1).unsqueeze_dim::<3>(2);
                    cdf.lower(uniform.unsqueeze_dim::<3>(1))
                        .int()
                        .cast(burn::tensor::DType::I64)
                        .sum_dim(1)
                        .squeeze_dim::<2>(1)
                        .clamp(0, categories as i64 - 1)
                        .cast(burn::tensor::DType::I64)
                }
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
    #[test]
    fn half_logits_promoted() {
        let node = MultinomialNodeBuilder::new("multinomial")
            .input_tensor("input", 2, DType::F16)
            .output_tensor("output", 2, DType::I32)
            .config(MultinomialConfig::default())
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2, Int> {
            let output = {
                let logits = input.cast(burn::tensor::DType::F32);
                if logits.dims()[0] == 0 {
                    burn::tensor::Tensor::<
                        2,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::new(
                            alloc::vec::Vec::<i64>::new(),
                            [0, 1usize],
                        ),
                        (&logits.device(), burn::tensor::DType::I32),
                    )
                } else {
                    let max = logits.clone().max_dim(1);
                    let weights = (logits - max).exp();
                    weights.categorical(1usize).cast(burn::tensor::DType::I32)
                }
            };
            output
        }
        ");
    }
}
