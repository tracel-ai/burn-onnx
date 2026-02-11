use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::deform_conv::DeformConvNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let weight_shape = self.inputs[1].ty.static_shape_known().unwrap_or_else(|| {
            panic!(
                "DeformConv '{}': weight tensor shape must be known at codegen time",
                self.name
            )
        });
        let groups = self.config.groups;
        let channels = [weight_shape[1] * groups, weight_shape[0]].to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let weight_groups = groups.to_tokens();
        let offset_groups = self.config.offset_groups.to_tokens();
        let padding = self.config.padding.to_tokens();

        // Bias is present if input[3] exists and is not optional
        let has_bias = self.inputs.get(3).is_some_and(|arg| !arg.is_optional());
        let bias = has_bias;

        Some(Field::new(
            self.name.clone(),
            quote! {
                DeformConv2d<B>
            },
            quote! {
                let #name = DeformConv2dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_weight_groups(#weight_groups)
                    .with_offset_groups(#offset_groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Weight tensor (input index 1)
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) = create_lazy_snapshot(weight_input, &weight_path, "DeformConv") {
                snapshots.push(snapshot);
            }
        }

        // Bias tensor (input index 3, optional)
        if let Some(bias_input) = self.inputs.get(3)
            && !bias_input.is_optional()
        {
            let bias_path = format!("{}.bias", field_name);
            if let Some(snapshot) = create_lazy_snapshot(bias_input, &bias_path, "DeformConv") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let offset = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        // Check if mask (input[4]) is present and not optional
        let has_mask = self.inputs.get(4).is_some_and(|arg| !arg.is_optional());

        if has_mask {
            let mask = scope.arg(&self.inputs[4]);
            quote! {
                let #output = self.#field.forward(#input, #offset, Some(#mask));
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input, #offset, None);
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::conv::DeformConv2d");
        imports.register("burn::nn::conv::DeformConv2dConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::deform_conv::{DeformConvConfig, DeformConvNode, DeformConvNodeBuilder};
    use onnx_ir::padding::PaddingConfig2d;

    fn create_deform_conv_node(name: &str, has_bias: bool, has_mask: bool) -> DeformConvNode {
        use onnx_ir::Argument;
        use onnx_ir::ir::{ArgType, TensorType};

        let config = DeformConvConfig::new(
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 1, 1, 1),
            [1, 1],
            1,
            1,
        );

        // Build the node with the required inputs first
        let mut node = DeformConvNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_tensor("offset", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();

        // Add bias or optional placeholder
        if has_bias {
            let mut arg = Argument::new(
                "bias",
                ArgType::Tensor(TensorType::new_known(DType::F32, vec![64])),
            );
            arg.value_source = onnx_ir::ir::ValueSource::Static(0);
            node.inputs.push(arg);
        } else {
            node.inputs.push(Argument::new("", ArgType::default()));
        }

        if has_mask {
            node.inputs.push(Argument::new(
                "mask",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 4,
                    static_shape: None,
                }),
            ));
        }

        node
    }

    #[test]
    fn test_deform_conv_field_init_with_bias() {
        let node = create_deform_conv_node("deform_conv1", true, false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let deform_conv1 = DeformConv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_weight_groups(1)
            .with_offset_groups(1)
            .with_bias(true)
            .init(device);
        ");
    }

    #[test]
    fn test_deform_conv_field_init_without_bias() {
        let node = create_deform_conv_node("deform_conv1", false, false);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let deform_conv1 = DeformConv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_weight_groups(1)
            .with_offset_groups(1)
            .with_bias(false)
            .init(device);
        ");
    }

    #[test]
    fn test_deform_conv_forward_without_mask() {
        let node = create_deform_conv_node("deform_conv1", true, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>, offset: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.deform_conv1.forward(input, offset, None);
            output
        }
        ");
    }

    #[test]
    fn test_deform_conv_forward_with_mask() {
        let node = create_deform_conv_node("deform_conv1", true, true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            offset: Tensor<B, 4>,
            mask: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let output = self.deform_conv1.forward(input, offset, Some(mask));
            output
        }
        ");
    }

    #[test]
    fn test_deform_conv_field_init_non_default_groups() {
        use onnx_ir::Argument;
        use onnx_ir::ir::ArgType;

        let config = DeformConvConfig::new([3, 3], [2, 2], PaddingConfig2d::Valid, [2, 2], 2, 4);

        let mut node = DeformConvNodeBuilder::new("deform_conv1")
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_tensor("offset", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();

        // Add optional bias placeholder
        node.inputs.push(Argument::new("", ArgType::default()));

        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let deform_conv1 = DeformConv2dConfig::new([6, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([2, 2])
            .with_weight_groups(2)
            .with_offset_groups(4)
            .with_bias(false)
            .init(device);
        ");
    }

    #[test]
    fn test_deform_conv_collect_snapshots_with_bias() {
        use crate::burn::node_traits::NodeCodegen;

        let node = create_deform_conv_node("deform_conv1", true, false);
        let snapshots = node.collect_snapshots("deform_conv1");
        // Weight + bias = 2 snapshots
        assert_eq!(snapshots.len(), 2);
    }

    #[test]
    fn test_deform_conv_collect_snapshots_without_bias() {
        use crate::burn::node_traits::NodeCodegen;

        let node = create_deform_conv_node("deform_conv1", false, false);
        let snapshots = node.collect_snapshots("deform_conv1");
        // Weight only = 1 snapshot
        assert_eq!(snapshots.len(), 1);
    }
}
