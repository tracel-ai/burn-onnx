use super::prelude::*;

impl NodeCodegen for onnx_ir::lp_pool1d::LpPool1dNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let kernel_size = self.config.kernel_size.to_tokens();
        let strides = self.config.stride.to_tokens();
        let ceil_mode = self.config.ceil_mode;

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_1d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial.as_deref(),
            self.config.kernel_size,
            self.config.stride,
            self.config.dilation,
        );

        Some(Field::new(
            self.name.clone(),
            quote! {
                AvgPool1d
            },
            quote! {
                let #name = AvgPool1dConfig::new(#kernel_size)
                    .with_stride(#strides)
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

        // onnx-ir only checks that p is finite and > 0, and the config is public, so p
        // or its reciprocal can still fall outside f32 (opset 1 allows p = 1e-39). A
        // non-finite literal would panic inside proc-macro2, so emit a named
        // `compile_error!` instead of crashing model generation.
        let p = self.config.p as f32;
        let p_inv = 1.0f32 / p;
        if !p.is_finite() || p <= 0.0 || !p_inv.is_finite() {
            let msg = format!(
                "LpPool1d node '{}': p must be > 0 with p and 1/p finite in f32, got {:?}",
                self.name, self.config.p
            );
            return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
        }
        let kernel_size = self.config.kernel_size as f32;

        quote! {
            let #output = self
                .#field
                .forward(#input.abs().powf_scalar(#p))
                .mul_scalar(#kernel_size)
                .powf_scalar(#p_inv);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::pool::AvgPool1d");
        imports.register("burn::nn::pool::AvgPool1dConfig");
        imports.register("burn::nn::PaddingConfig1d");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::lp_pool1d::{LpPool1dConfig, LpPool1dNode, LpPool1dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig1d};

    fn create_lp_pool1d_node(name: &str, p: f64) -> LpPool1dNode {
        let config = LpPool1dConfig::new(
            3,
            2,
            PaddingConfig1d::Explicit(1, 1),
            1,
            false,
            AutoPad::NotSet,
            p,
        );

        LpPool1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_lp_pool1d_forward() {
        let node = create_lp_pool1d_node("pool1", 3.0);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self
                .pool1
                .forward(input.abs().powf_scalar(3f32))
                .mul_scalar(3f32)
                .powf_scalar(0.33333334f32);
            output
        }
        ");
    }

    #[test]
    fn test_lp_pool1d_forward_fractional_p() {
        let node = create_lp_pool1d_node("pool1", 1.5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self
                .pool1
                .forward(input.abs().powf_scalar(1.5f32))
                .mul_scalar(3f32)
                .powf_scalar(0.6666667f32);
            output
        }
        ");
    }

    #[test]
    fn test_lp_pool1d_tiny_p_emits_compile_error() {
        // 1/p overflows f32 even though p itself is a finite positive f32.
        let node = create_lp_pool1d_node("pool1", 1e-39);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                compile_error!(
                    "LpPool1d node 'pool1': p must be > 0 with p and 1/p finite in f32, got 1e-39"
                );
                unreachable!()
            };
            output
        }
        "#);
    }

    #[test]
    fn test_lp_pool1d_non_finite_p_emits_compile_error() {
        let node = create_lp_pool1d_node("pool1", f64::NAN);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                compile_error!(
                    "LpPool1d node 'pool1': p must be > 0 with p and 1/p finite in f32, got NaN"
                );
                unreachable!()
            };
            output
        }
        "#);
    }

    #[test]
    fn test_lp_pool1d_forward_with_clone() {
        let node = create_lp_pool1d_node("pool1", 3.0);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self
                .pool1
                .forward(input.clone().abs().powf_scalar(3f32))
                .mul_scalar(3f32)
                .powf_scalar(0.33333334f32);
            output
        }
        ");
    }

    #[test]
    fn test_lp_pool1d_field_init() {
        let node = create_lp_pool1d_node("pool1", 3.0);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r#"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_count_include_pad(true)
            .with_ceil_mode(false)
            .init();
        "#);
    }

    #[test]
    fn test_lp_pool1d_field_init_auto_pad_same_upper_dynamic() {
        let config = LpPool1dConfig::new(
            3,
            1,
            PaddingConfig1d::Valid,
            1,
            false,
            AutoPad::SameUpper,
            2.0,
        );
        let node = LpPool1dNodeBuilder::new("pool1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        // Dynamic length: burn computes the pads at forward time.
        assert_snapshot!(code, @r"
        let pool1 = AvgPool1dConfig::new(3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same)
            .with_count_include_pad(true)
            .with_ceil_mode(false)
            .init();
        ");
    }
}
