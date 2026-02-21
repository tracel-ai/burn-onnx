use super::prelude::*;

impl NodeCodegen for onnx_ir::lp_pool2d::LpPool2dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.strides.to_tokens();
        let ceil_mode = self.config.ceil_mode;

        let shape = self.inputs[0].ty.static_shape_known();
        let input_spatial = shape.as_deref().map(|s| &s[2..]);
        let padding = crate::burn::codegen::resolve_auto_pad_2d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial,
            &self.config.kernel_size,
            &self.config.strides,
            &self.config.dilation,
        )
        .to_tokens();

        Some(Field::new(
            self.name.clone(),
            quote! {
                AvgPool2d
            },
            quote! {
                let #name = AvgPool2dConfig::new(#kernel_size)
                    .with_strides(#strides)
                    .with_padding(#padding)
                    .with_count_include_pad(true)
                    .with_ceil_mode(#ceil_mode)
                    .init();
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        let p = self.config.p as f32;
        let p_inv = 1.0f32 / p;
        let kernel_size = (self.config.kernel_size[0] * self.config.kernel_size[1]) as f32;

        quote! {
            let #output = self
                .#field
                .forward(#input.abs().powf_scalar(#p))
                .mul_scalar(#kernel_size)
                .powf_scalar(#p_inv);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::pool::AvgPool2d");
        imports.register("burn::nn::pool::AvgPool2dConfig");
        imports.register("burn::nn::PaddingConfig2d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::lp_pool2d::{LpPool2dConfig, LpPool2dNode, LpPool2dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig2d};

    fn create_lp_pool2d_node(name: &str, p: i64) -> LpPool2dNode {
        let config = LpPool2dConfig::new(
            [2, 3],
            [1, 2],
            PaddingConfig2d::Explicit(0, 1, 1, 0),
            [1, 1],
            false,
            AutoPad::NotSet,
            p,
        );

        LpPool2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_lp_pool2d_forward() {
        let node = create_lp_pool2d_node("pool1", 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self
                .pool1
                .forward(input.abs().powf_scalar(2f32))
                .mul_scalar(6f32)
                .powf_scalar(0.5f32);
            output
        }
        ");
    }

    #[test]
    fn test_lp_pool2d_field_init() {
        let node = create_lp_pool2d_node("pool1", 2);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool2dConfig::new([2, 3])
            .with_strides([1, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 1, 0))
            .with_count_include_pad(true)
            .with_ceil_mode(false)
            .init();
        "#);
    }
}
