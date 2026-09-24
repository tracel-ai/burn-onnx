use super::prelude::*;
use burn_pack::Tensor as PackTensor;

impl NodeCodegen for onnx_ir::conv3d::Conv3dNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !self.inputs[1].is_static() {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let weight_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("Conv3d: weight tensor shape must be known at codegen time");
        let groups = self.config.groups;
        let channels = [weight_shape[1] * groups, weight_shape[0]].to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = groups.to_tokens();
        let bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_3d(
            &self.config.auto_pad,
            &self.config.padding,
            input_spatial.as_deref(),
            &self.config.kernel_size,
            &self.config.stride,
            &self.config.dilation,
        );

        Some(Field::new(
            self.name.clone(),
            quote! {
                Conv3d
            },
            quote! {
                let #name = Conv3dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // A runtime weight has no module to live in, so the functional op takes it.
        if !self.inputs[1].is_static() {
            let (front, top, left, back, bottom, right) = self.config.padding.as_tuple();
            let explicit = [(front, back), (top, bottom), (left, right)];
            let geometry = super::conv_helpers::ConvGeometry {
                auto_pad: &self.config.auto_pad,
                explicit: &explicit,
                kernel: &self.config.kernel_size,
                stride: &self.config.stride,
                dilation: &self.config.dilation,
                groups: self.config.groups,
            };
            return super::conv_helpers::functional_conv(
                scope,
                &self.inputs,
                &self.outputs[0],
                "conv3d",
                geometry,
            );
        }
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if !self.inputs[1].is_static() {
            return;
        }
        imports.register("burn::nn::PaddingConfig3d");
        imports.register("burn::nn::conv::Conv3d");
        imports.register("burn::nn::conv::Conv3dConfig");
    }

    fn collect_tensors(&self, field_name: &str) -> Vec<PackTensor> {
        if !self.inputs[1].is_static() {
            return vec![];
        }
        use crate::burn::node_traits::create_deferred_tensor;
        let mut tensors = vec![];

        // Weight tensor (input index 1)
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(tensor) = create_deferred_tensor(weight_input, &weight_path) {
                tensors.push(tensor);
            }
        }

        // Bias tensor if present (input index 2)
        if let Some(bias_input) = self.inputs.get(2) {
            let bias_path = format!("{}.bias", field_name);
            if let Some(tensor) = create_deferred_tensor(bias_input, &bias_path) {
                tensors.push(tensor);
            }
        }

        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::conv3d::{Conv3dConfig, Conv3dNode, Conv3dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig3d};

    fn create_conv3d_node(name: &str) -> Conv3dNode {
        let config = Conv3dConfig::new(
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            1,
            PaddingConfig3d::Explicit(1, 1, 1, 1, 1, 1),
            AutoPad::NotSet,
        );

        Conv3dNodeBuilder::new(name)
            .input_tensor("input", 5, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build()
    }

    fn create_conv3d_node_asymmetric(name: &str) -> Conv3dNode {
        let config = Conv3dConfig::new(
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            1,
            PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6),
            AutoPad::NotSet,
        );

        Conv3dNodeBuilder::new(name)
            .input_tensor("input", 5, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv3d_forward() {
        let node = create_conv3d_node("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = self.conv1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv3d_forward_with_clone() {
        let node = create_conv3d_node("conv1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = self.conv1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_conv3d_field_init_auto_pad_same_upper() {
        let config = Conv3dConfig::new(
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            1,
            PaddingConfig3d::Valid,
            AutoPad::SameUpper,
        );
        let node = Conv3dNodeBuilder::new("conv1")
            .input_tensor_shape("input", vec![1, 3, 7, 7, 7], DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let conv1 = Conv3dConfig::new([3, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    #[should_panic(expected = "Asymmetric 3D padding is not supported by Burn")]
    fn test_conv3d_field_init_asymmetric_padding() {
        let node = create_conv3d_node_asymmetric("conv1");
        // Asymmetric 3D padding panics at codegen time since Burn doesn't support it
        let _ = codegen_field_init(&node);
    }
    #[test]
    fn test_conv3d_runtime_weight() {
        let node = {
            let config = Conv3dConfig::new(
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                1,
                PaddingConfig3d::Explicit(1, 1, 1, 1, 1, 1),
                AutoPad::NotSet,
            );

            Conv3dNodeBuilder::new("conv1")
                .input_tensor("input", 5, DType::F32)
                .input_tensor("weight", 5, DType::F32)
                .input_tensor("bias", 1, DType::F32)
                .output_tensor("output", 5, DType::F32)
                .config(config)
                .build()
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<5>,
            weight: Tensor<5>,
            bias: Tensor<1>,
        ) -> Tensor<5> {
            let output = burn::tensor::module::conv3d(
                input,
                weight,
                Some(bias),
                burn::tensor::ops::ConvOptions::new_with_padding(
                    [1, 1, 1],
                    [(1usize, 1usize), (1usize, 1usize), (1usize, 1usize)],
                    [1, 1, 1],
                    1,
                ),
            );
            output
        }
        ");
    }
}
