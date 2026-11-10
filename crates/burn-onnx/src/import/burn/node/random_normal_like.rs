use super::prelude::*;

impl NodeCodegen for onnx_ir::node::random_like::RandomNormalLikeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Build distribution with mean and scale (standard deviation)
        let mean = self.config.mean;
        let std_deviation = self.config.scale;
        let dist = quote! { Distribution::Normal(#mean, #std_deviation) };

        let dtype = self.outputs[0].ty.elem_type().to_tokens();
        let draw = super::random_helpers::random_draw(
            self.config.seed,
            quote! { input.shape() },
            dist,
            quote! { &input.device() },
            dtype,
        );
        quote! { let #output = { let input = #input; #draw }; }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::Distribution");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::random_like::{RandomNormalLikeConfig, RandomNormalLikeNodeBuilder};

    #[test]
    fn test_random_normal_like_seeded() {
        let config = RandomNormalLikeConfig {
            seed: Some(42.5),
            mean: 0.0,
            scale: 1.0,
        };
        let node = RandomNormalLikeNodeBuilder::new("randl1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = {
                let input = input;
                Tensor::from_dispatch(
                    <burn::backend::Dispatch as burn::backend::ops::FloatTensorOps<
                        burn::backend::Dispatch,
                    >>::float_random(
                        (input.shape()).into(),
                        Distribution::Normal(0f64, 1f64),
                        (&input.device()).as_dispatch(),
                        (burn::tensor::DType::F64).into(),
                        burn::backend::RandomKey::new(1110048768u64),
                    ),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_random_normal_like() {
        let config = RandomNormalLikeConfig {
            seed: None,
            mean: 0.0,
            scale: 1.0,
        };
        let node = RandomNormalLikeNodeBuilder::new("randl1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = {
                let input = input;
                Tensor::random(
                    input.shape(),
                    Distribution::Normal(0f64, 1f64),
                    (&input.device(), burn::tensor::DType::F32),
                )
            };
            output
        }
        ");
    }
}
