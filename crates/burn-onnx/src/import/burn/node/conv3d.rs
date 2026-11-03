use super::prelude::*;
use burn_pack::Tensor as PackTensor;
use onnx_ir::conv3d::Conv3dNode;

impl NodeCodegen for Conv3dNode {
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

        // Pads burn-nn cannot express are applied by the functional op in `forward`, so the
        // module itself pads nothing and only holds the weight and bias.
        let padding = module_padding(self).unwrap_or_else(|| quote! { PaddingConfig3d::Valid });

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
        let explicit = explicit_pads(self);
        // A runtime weight has no module to live in, so the functional op takes it.
        if !self.inputs[1].is_static() {
            return super::conv_helpers::functional_conv(
                scope,
                &self.inputs,
                &self.outputs[0],
                "conv3d",
                geometry(self, &explicit),
            );
        }
        let input = scope.arg(self.inputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if module_padding(self).is_none() {
            // The module's weight and bias, run through the functional op that takes
            // `(begin, end)` pads.
            let operands = super::conv_helpers::ConvOperands {
                input,
                weight: quote! { self.#field.weight.val() },
                bias: quote! { self.#field.bias.as_ref().map(burn::module::Param::val) },
            };
            return super::conv_helpers::functional_conv_call(
                &self.inputs[0],
                operands,
                &self.outputs[0],
                "conv3d",
                geometry(self, &explicit),
            );
        }

        let output = arg_to_ident(self.outputs.first().unwrap());
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

/// Explicit `pads` as `(begin, end)` per spatial axis: depth, height, width.
fn explicit_pads(node: &Conv3dNode) -> [(usize, usize); 3] {
    let (front, top, left, back, bottom, right) = node.config.padding.as_tuple();
    [(front, back), (top, bottom), (left, right)]
}

/// The conv's geometry for the functional op, with `explicit` from [`explicit_pads`].
fn geometry<'a>(
    node: &'a Conv3dNode,
    explicit: &'a [(usize, usize)],
) -> super::conv_helpers::ConvGeometry<'a> {
    super::conv_helpers::ConvGeometry {
        auto_pad: &node.config.auto_pad,
        explicit,
        kernel: &node.config.kernel_size,
        stride: &node.config.stride,
        dilation: &node.config.dilation,
        groups: node.config.groups,
    }
}

/// The burn-nn `PaddingConfig3d` the `Conv3d` module field pads with, or `None` when the
/// functional op has to apply the pads instead: burn-nn's explicit 3D padding is symmetric,
/// so this covers pads that differ between the two ends of an axis (explicit, or SAME with an
/// odd total), as well as SAME on an input sized only at run time.
fn module_padding(node: &Conv3dNode) -> Option<TokenStream> {
    let explicit = explicit_pads(node);
    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&node.inputs[0].ty);
    let pads = crate::burn::codegen::resolve_padding_pairs(
        &node.config.auto_pad,
        &explicit,
        input_spatial.as_deref(),
        &node.config.kernel_size,
        &node.config.stride,
        &node.config.dilation,
    )?;
    crate::burn::codegen::padding_config_3d_tokens(&pads)
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
    fn test_conv3d_field_init_asymmetric_padding() {
        // burn-nn cannot express the pads, so the module pads nothing.
        let node = create_conv3d_node_asymmetric("conv1");
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let conv1 = Conv3dConfig::new([3, 64], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_conv3d_forward_asymmetric_padding() {
        let node = create_conv3d_node_asymmetric("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = burn::tensor::module::conv3d(
                input,
                self.conv1.weight.val(),
                self.conv1.bias.as_ref().map(burn::module::Param::val),
                burn::tensor::ops::ConvOptions::new_with_padding(
                    [1, 1, 1],
                    [(1usize, 4usize), (2usize, 5usize), (3usize, 6usize)],
                    [1, 1, 1],
                    1,
                ),
            );
            output
        }
        ");
    }

    #[test]
    fn test_conv3d_forward_auto_pad_same_upper_asymmetric() {
        // An even kernel extent leaves an odd SAME total on depth and width.
        let config = Conv3dConfig::new(
            [2, 3, 2],
            [1, 1, 1],
            [1, 1, 1],
            1,
            PaddingConfig3d::Valid,
            AutoPad::SameUpper,
        );
        let node = Conv3dNodeBuilder::new("conv1")
            .input_tensor_shape("input", vec![1, 3, 7, 7, 7], DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 2, 3, 2], DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build();
        assert_snapshot!(codegen_field_init(&node), @r"
        let conv1 = Conv3dConfig::new([3, 64], [2, 3, 2])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        ");
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = burn::tensor::module::conv3d(
                input,
                self.conv1.weight.val(),
                self.conv1.bias.as_ref().map(burn::module::Param::val),
                burn::tensor::ops::ConvOptions::new_with_padding(
                    [1, 1, 1],
                    [(0usize, 1usize), (1usize, 1usize), (0usize, 1usize)],
                    [1, 1, 1],
                    1,
                ),
            );
            output
        }
        ");
    }

    #[test]
    fn test_conv3d_runtime_weight_asymmetric_padding() {
        let config = Conv3dConfig::new(
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            1,
            PaddingConfig3d::Explicit(0, 1, 0, 1, 0, 1),
            AutoPad::NotSet,
        );
        let node = Conv3dNodeBuilder::new("conv1")
            .input_tensor("input", 5, DType::F32)
            .input_tensor("weight", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>, weight: Tensor<5>) -> Tensor<5> {
            let output = burn::tensor::module::conv3d(
                input,
                weight,
                None,
                burn::tensor::ops::ConvOptions::new_with_padding(
                    [1, 1, 1],
                    [(0usize, 1usize), (1usize, 0usize), (0usize, 1usize)],
                    [1, 1, 1],
                    1,
                ),
            );
            output
        }
        ");
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
