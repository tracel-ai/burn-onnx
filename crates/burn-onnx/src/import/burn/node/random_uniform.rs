use super::prelude::*;

impl NodeCodegen for onnx_ir::node::random::RandomUniformNode {
    fn inputs(&self) -> &[Argument] {
        // RandomUniform has no inputs - it generates a tensor from scratch
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Build shape expression
        let shape_values = self.config.shape.iter();
        let shape = quote! { Shape::new([#(#shape_values),*]) };

        // Build distribution with low and high bounds
        let low = self.config.low;
        let high = self.config.high;
        let dist = quote! { Distribution::Uniform(#low, #high) };

        let dtype = self.outputs[0].ty.elem_type().to_tokens();
        let draw = super::random_helpers::random_draw(
            self.config.seed,
            shape,
            dist,
            quote! { self.device() },
            dtype,
        );
        quote! { let #output = #draw; }
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
    use onnx_ir::node::random::{RandomUniformConfig, RandomUniformNodeBuilder};

    #[test]
    fn test_random_uniform_seeded() {
        let config = RandomUniformConfig {
            seed: Some(42.5),
            low: 0.0,
            high: 1.0,
            shape: vec![3, 4],
        };
        let node = RandomUniformNodeBuilder::new("rand1")
            .output_tensor("output", 2, DType::F64)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self) -> Tensor<2> {
            let output = Tensor::from_dispatch(
                <burn::backend::Dispatch as burn::backend::ops::FloatTensorOps<
                    burn::backend::Dispatch,
                >>::float_random(
                    (Shape::new([3usize, 4usize])).into(),
                    Distribution::Uniform(0f64, 1f64),
                    (self.device()).as_dispatch(),
                    (burn::tensor::DType::F64).into(),
                    burn::backend::RandomKey::new(1110048768u64),
                ),
            );
            output
        }
        ");
    }

    #[test]
    fn test_random_uniform() {
        let config = RandomUniformConfig {
            seed: None,
            low: 0.0,
            high: 1.0,
            shape: vec![3, 4],
        };
        let node = RandomUniformNodeBuilder::new("rand1")
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<2> {
            let output = Tensor::random(
                Shape::new([3usize, 4usize]),
                Distribution::Uniform(0f64, 1f64),
                (self.device(), burn::tensor::DType::F32),
            );
            output
        }
        ");
    }
}
