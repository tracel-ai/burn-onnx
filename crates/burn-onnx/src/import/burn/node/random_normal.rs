use super::prelude::*;

impl NodeCodegen for onnx_ir::node::random::RandomNormalNode {
    fn inputs(&self) -> &[Argument] {
        // RandomNormal has no inputs - it generates a tensor from scratch
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

        // Build distribution with mean and scale (standard deviation)
        let mean = self.config.mean;
        let std_deviation = self.config.scale;
        let dist = quote! { Distribution::Normal(#mean, #std_deviation) };

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
    use onnx_ir::node::random::{RandomNormalConfig, RandomNormalNodeBuilder};

    #[test]
    fn test_random_normal_seeded() {
        let config = RandomNormalConfig {
            seed: Some(42.5),
            mean: 0.0,
            scale: 1.0,
            shape: vec![2, 3, 4],
        };
        let node = RandomNormalNodeBuilder::new("rand1")
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self) -> Tensor<3> {
            let output = Tensor::from_dispatch(
                <burn::backend::Dispatch as burn::backend::ops::FloatTensorOps<
                    burn::backend::Dispatch,
                >>::float_random(
                    (Shape::new([2usize, 3usize, 4usize])).into(),
                    Distribution::Normal(0f64, 1f64),
                    (self.device()).as_dispatch(),
                    (burn::tensor::DType::F32).into(),
                    burn::backend::RandomKey::new(1110048768u64),
                ),
            );
            output
        }
        ");
    }

    #[test]
    fn test_random_normal() {
        let config = RandomNormalConfig {
            seed: None,
            mean: 0.0,
            scale: 1.0,
            shape: vec![2, 3, 4],
        };
        let node = RandomNormalNodeBuilder::new("rand1")
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<3> {
            let output = Tensor::random(
                Shape::new([2usize, 3usize, 4usize]),
                Distribution::Normal(0f64, 1f64),
                (self.device(), burn::tensor::DType::F32),
            );
            output
        }
        ");
    }
}
