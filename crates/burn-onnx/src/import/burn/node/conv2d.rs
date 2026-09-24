use super::prelude::*;
use burn_pack::Tensor as PackTensor;

impl NodeCodegen for onnx_ir::conv2d::Conv2dNode {
    fn inputs(&self) -> &[Argument] {
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
            .expect("Conv2d: weight tensor shape must be known at codegen time");
        let groups = self.config.groups;
        let channels = [weight_shape[1] * groups, weight_shape[0]].to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = groups.to_tokens();
        let bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

        let input_spatial = onnx_ir::node::padding::static_spatial_dims(&self.inputs[0].ty);
        let padding = crate::burn::codegen::resolve_auto_pad_2d(
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
                Conv2d
            },
            quote! {
                let #name = Conv2dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
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

        // Bias tensor (input index 2, optional)
        if self.inputs.len() > 2
            && let Some(bias_input) = self.inputs.get(2)
        {
            let bias_path = format!("{}.bias", field_name);
            if let Some(tensor) = create_deferred_tensor(bias_input, &bias_path) {
                tensors.push(tensor);
            }
        }

        tensors
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        // A runtime weight has no module to live in, so the functional op takes it.
        if !self.inputs[1].is_static() {
            let (top, left, bottom, right) = self.config.padding.as_tuple();
            let explicit = [(top, bottom), (left, right)];
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
                "conv2d",
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
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::conv::Conv2d");
        imports.register("burn::nn::conv::Conv2dConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::conv2d::{Conv2dConfig, Conv2dNode, Conv2dNodeBuilder};
    use onnx_ir::padding::{AutoPad, PaddingConfig2d};

    fn create_conv2d_node(name: &str) -> Conv2dNode {
        let config = Conv2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 1, 1, 1),
            [1, 1],
            1,
            AutoPad::NotSet,
        );

        Conv2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_conv2d_node_asymmetric(name: &str) -> Conv2dNode {
        // Asymmetric padding: top=1, left=2, bottom=3, right=4
        let config = Conv2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 2, 3, 4),
            [1, 1],
            1,
            AutoPad::NotSet,
        );

        Conv2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv2d_forward() {
        let node = create_conv2d_node("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.conv1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv2d_forward_with_clone() {
        let node = create_conv2d_node("conv1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.conv1.forward(input.clone());
            output
        }
        ");
    }

    fn create_conv2d_node_auto_pad(name: &str, auto_pad: AutoPad) -> Conv2dNode {
        let config = Conv2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid, // ignored when auto_pad is set
            [1, 1],
            1,
            auto_pad,
        );

        Conv2dNodeBuilder::new(name)
            .input_tensor_shape("input", vec![1, 3, 7, 7], DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv2d_field_init_auto_pad_same_upper() {
        let node = create_conv2d_node_auto_pad("conv1", AutoPad::SameUpper);
        let code = codegen_field_init(&node);
        // auto_pad SAME_UPPER with input=7x7, kernel=3x3, stride=1 -> pad 1 each side
        assert_snapshot!(code, @r"
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv2d_field_init_auto_pad_same_upper_dynamic() {
        let config = Conv2dConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Valid,
            [1, 1],
            1,
            AutoPad::SameUpper,
        );
        let node = Conv2dNodeBuilder::new("conv1")
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_field_init(&node);
        // Dynamic H/W: burn computes the pads at forward time.
        assert_snapshot!(code, @r"
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv2d_field_init_auto_pad_valid() {
        let node = create_conv2d_node_auto_pad("conv1", AutoPad::Valid);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv2d_field_init_asymmetric_padding() {
        let node = create_conv2d_node_asymmetric("conv1");
        let code = codegen_field_init(&node);
        // Asymmetric padding is passed directly to the module
        assert_snapshot!(code, @r"
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 2, 3, 4))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }
    #[test]
    fn test_conv2d_runtime_weight() {
        let node = {
            let config = Conv2dConfig::new(
                [3, 3],
                [1, 1],
                PaddingConfig2d::Explicit(1, 1, 1, 1),
                [1, 1],
                1,
                AutoPad::NotSet,
            );

            Conv2dNodeBuilder::new("conv1")
                .input_tensor("input", 4, DType::F32)
                .input_tensor("weight", 4, DType::F32)
                .input_tensor("bias", 1, DType::F32)
                .output_tensor("output", 4, DType::F32)
                .config(config)
                .build()
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<4>,
            weight: Tensor<4>,
            bias: Tensor<1>,
        ) -> Tensor<4> {
            let output = burn::tensor::module::conv2d(
                input,
                weight,
                Some(bias),
                burn::tensor::ops::ConvOptions::new_with_padding(
                    [1, 1],
                    [(1usize, 1usize), (1usize, 1usize)],
                    [1, 1],
                    1,
                ),
            );
            output
        }
        ");
    }

    #[test]
    fn test_conv2d_runtime_weight_dynamic_same_padding() {
        let config = Conv2dConfig::new(
            [2, 2],
            [1, 1],
            PaddingConfig2d::Valid,
            [1, 1],
            1,
            AutoPad::SameUpper,
        );
        let node = Conv2dNodeBuilder::new("conv1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("weight", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, weight: Tensor<4>) -> Tensor<4> {
            let output = {
                let padding = {
                    let dims = input.dims();
                    [
                        {
                            let size = dims[2usize];
                            let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                + 2usize)
                                .saturating_sub(size);
                            let small = total / 2;
                            let big = total - small;
                            (small, big)
                        },
                        {
                            let size = dims[3usize];
                            let total = (size.div_ceil(1usize).saturating_sub(1) * 1usize
                                + 2usize)
                                .saturating_sub(size);
                            let small = total / 2;
                            let big = total - small;
                            (small, big)
                        },
                    ]
                };
                burn::tensor::module::conv2d(
                    input,
                    weight,
                    None,
                    burn::tensor::ops::ConvOptions::new_with_padding([1, 1], padding, [1, 1], 1),
                )
            };
            output
        }
        ");
    }
}
