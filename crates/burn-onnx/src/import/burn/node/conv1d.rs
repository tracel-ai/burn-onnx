use super::prelude::*;
use burn_pack::Tensor as PackTensor;

impl NodeCodegen for onnx_ir::conv1d::Conv1dNode {
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
            .expect("Conv1d: weight tensor shape must be known at codegen time");
        let groups = self.config.groups;
        let channels_in = (weight_shape[1] * groups).to_tokens();
        let channels_out = weight_shape[0].to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = groups.to_tokens();
        let bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

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
                Conv1d
            },
            quote! {
                let #name = Conv1dConfig::new(#channels_in, #channels_out, #kernel_size)
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
            let (left, right) = self.config.padding.as_tuple();
            let explicit = [(left, right)];
            let geometry = super::conv_helpers::ConvGeometry {
                auto_pad: &self.config.auto_pad,
                explicit: &explicit,
                kernel: &[self.config.kernel_size],
                stride: &[self.config.stride],
                dilation: &[self.config.dilation],
                groups: self.config.groups,
            };
            return super::conv_helpers::functional_conv(
                scope,
                &self.inputs,
                &self.outputs[0],
                "conv1d",
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
        imports.register("burn::nn::PaddingConfig1d");
        imports.register("burn::nn::conv::Conv1d");
        imports.register("burn::nn::conv::Conv1dConfig");
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
    use onnx_ir::conv1d::{Conv1dConfig, Conv1dNode, Conv1dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig1d};

    fn create_conv1d_node(name: &str) -> Conv1dNode {
        let config =
            Conv1dConfig::new(3, 1, 1, 1, PaddingConfig1d::Explicit(1, 1), AutoPad::NotSet);

        Conv1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    fn create_conv1d_node_asymmetric(name: &str) -> Conv1dNode {
        // Asymmetric padding: left=1, right=2
        let config =
            Conv1dConfig::new(3, 1, 1, 1, PaddingConfig1d::Explicit(1, 2), AutoPad::NotSet);

        Conv1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv1d_forward() {
        let node = create_conv1d_node("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.conv1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv1d_forward_with_clone() {
        let node = create_conv1d_node("conv1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.conv1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_conv1d_field_init_auto_pad_same_upper() {
        let config = Conv1dConfig::new(3, 1, 1, 1, PaddingConfig1d::Valid, AutoPad::SameUpper);
        let node = Conv1dNodeBuilder::new("conv1")
            .input_tensor_shape("input", vec![1, 3, 7], DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let conv1 = Conv1dConfig::new(3, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv1d_field_init_auto_pad_same_upper_dynamic() {
        let config = Conv1dConfig::new(3, 1, 1, 1, PaddingConfig1d::Valid, AutoPad::SameUpper);
        let node = Conv1dNodeBuilder::new("conv1")
            .input_tensor("input", 3, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        // Dynamic length: burn computes the pads at forward time.
        assert_snapshot!(code, @r"
        let conv1 = Conv1dConfig::new(3, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Same)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv1d_field_init_asymmetric_padding() {
        let node = create_conv1d_node_asymmetric("conv1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let conv1 = Conv1dConfig::new(3, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 2))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }
    #[test]
    fn test_conv1d_runtime_weight() {
        let node = {
            let config =
                Conv1dConfig::new(3, 1, 1, 1, PaddingConfig1d::Explicit(1, 1), AutoPad::NotSet);

            Conv1dNodeBuilder::new("conv1")
                .input_tensor("input", 3, DType::F32)
                .input_tensor("weight", 3, DType::F32)
                .input_tensor("bias", 1, DType::F32)
                .output_tensor("output", 3, DType::F32)
                .config(config)
                .build()
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            weight: Tensor<3>,
            bias: Tensor<1>,
        ) -> Tensor<3> {
            let output = burn::tensor::module::conv1d(
                input,
                weight,
                Some(bias),
                burn::tensor::ops::ConvOptions::new_with_padding([1], [(1usize, 1usize)], [1], 1),
            );
            output
        }
        ");
    }
}
