use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::deform_conv::DeformConvNode;
use onnx_ir::padding::PaddingConfig2d;

/// True when the weight (input[1]) was lifted to a static initializer. When
/// false, weight arrives as a runtime forward parameter and we have to use
/// the `deform_conv2d` function form instead of the `DeformConv2d` module.
fn weight_is_static(node: &DeformConvNode) -> bool {
    node.inputs.get(1).is_some_and(|w| w.is_static())
}

/// Convert `PaddingConfig2d` into a symmetric `[usize; 2]` for Burn's
/// `deform_conv2d`, or a `compile_error!` token if the padding is
/// asymmetric (which neither the function form nor the module form
/// supports today). Surfacing the failure as a generated `compile_error!`
/// keeps codegen panic-free; a future fix could emit an explicit Pad op
/// upstream instead.
fn padding_to_symmetric_tokens(padding: &PaddingConfig2d, node_name: &str) -> TokenStream {
    match padding {
        PaddingConfig2d::Valid => quote! { [0usize, 0usize] },
        PaddingConfig2d::Explicit(top, left, bottom, right) if top == bottom && left == right => {
            let t = top.to_tokens();
            let l = left.to_tokens();
            quote! { [#t, #l] }
        }
        PaddingConfig2d::Explicit(top, left, bottom, right) => {
            let msg = format!(
                "DeformConv '{node_name}': asymmetric padding ({top}, {left}, {bottom}, {right}) \
                 is not supported by Burn's deform_conv2d (symmetric only). \
                 Convert via an explicit Pad op upstream."
            );
            quote! { compile_error!(#msg) }
        }
    }
}

impl NodeCodegen for DeformConvNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !weight_is_static(self) {
            return None;
        }
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

        let has_bias = self.inputs.get(3).is_some_and(|arg| !arg.is_optional());
        let bias = has_bias;

        Some(Field::new(
            self.name.clone(),
            quote! { DeformConv2d<B> },
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

        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) = create_lazy_snapshot(weight_input, &weight_path, "DeformConv") {
                snapshots.push(snapshot);
            }
        }

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
        let output = arg_to_ident(self.outputs.first().unwrap());

        if weight_is_static(self) {
            let input = scope.arg(&self.inputs[0]);
            let offset = scope.arg(&self.inputs[2]);
            let field = Ident::new(&self.name, Span::call_site());
            let has_mask = self.inputs.get(4).is_some_and(|arg| !arg.is_optional());

            return if has_mask {
                let mask = scope.arg(&self.inputs[4]);
                quote! { let #output = self.#field.forward(#input, #offset, Some(#mask)); }
            } else {
                quote! { let #output = self.#field.forward(#input, #offset, None); }
            };
        }

        let input = scope.arg(&self.inputs[0]);
        let weight = scope.arg(&self.inputs[1]);
        let offset = scope.arg(&self.inputs[2]);

        let bias_token = match self.inputs.get(3) {
            Some(arg) if !arg.is_optional() => {
                let b = scope.arg(arg);
                quote! { Some(#b) }
            }
            _ => quote! { None },
        };
        let mask_token = match self.inputs.get(4) {
            Some(arg) if !arg.is_optional() => {
                let m = scope.arg(arg);
                quote! { Some(#m) }
            }
            _ => quote! { None },
        };

        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let weight_groups = self.config.groups.to_tokens();
        let offset_groups = self.config.offset_groups.to_tokens();
        let padding = padding_to_symmetric_tokens(&self.config.padding, &self.name);

        quote! {
            let #output = burn::tensor::module::deform_conv2d(
                #input,
                #offset,
                #weight,
                #mask_token,
                #bias_token,
                burn::tensor::ops::DeformConvOptions::new(
                    #stride,
                    #padding,
                    #dilation,
                    #weight_groups,
                    #offset_groups,
                ),
            );
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if weight_is_static(self) {
            imports.register("burn::nn::PaddingConfig2d");
            imports.register("burn::nn::conv::DeformConv2d");
            imports.register("burn::nn::conv::DeformConv2dConfig");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::deform_conv::{DeformConvConfig, DeformConvNode, DeformConvNodeBuilder};
    use onnx_ir::padding::PaddingConfig2d;

    fn create_static_deform_conv_node(
        name: &str,
        has_bias: bool,
        has_mask: bool,
    ) -> DeformConvNode {
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

        let mut node = DeformConvNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![64, 3, 3, 3], DType::F32)
            .input_tensor("offset", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();

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

    fn create_dynamic_deform_conv_node(
        name: &str,
        padding: PaddingConfig2d,
        has_bias: bool,
        has_mask: bool,
    ) -> DeformConvNode {
        use onnx_ir::Argument;
        use onnx_ir::ir::{ArgType, TensorType};

        let config = DeformConvConfig::new([2, 2], [1, 1], padding, [1, 1], 1, 1);

        let mut node = DeformConvNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("weight", vec![1, 1, 2, 2], DType::F32)
            .input_tensor("offset", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();

        if has_bias {
            node.inputs.push(Argument::new(
                "bias",
                ArgType::Tensor(TensorType::new_known(DType::F32, vec![1])),
            ));
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
        let node = create_static_deform_conv_node("deform_conv1", true, false);
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
        let node = create_static_deform_conv_node("deform_conv1", false, false);
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
        let node = create_static_deform_conv_node("deform_conv1", true, false);
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
        let node = create_static_deform_conv_node("deform_conv1", true, true);
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

        let node = create_static_deform_conv_node("deform_conv1", true, false);
        let snapshots = node.collect_snapshots("deform_conv1");
        assert_eq!(snapshots.len(), 2);
    }

    #[test]
    fn test_deform_conv_collect_snapshots_without_bias() {
        use crate::burn::node_traits::NodeCodegen;

        let node = create_static_deform_conv_node("deform_conv1", false, false);
        let snapshots = node.collect_snapshots("deform_conv1");
        assert_eq!(snapshots.len(), 1);
    }

    #[test]
    fn test_deform_conv_field_dynamic_emits_no_field() {
        let node =
            create_dynamic_deform_conv_node("deform_conv1", PaddingConfig2d::Valid, false, false);
        let code = codegen_field_init(&node);
        assert_eq!(code, "");
    }

    #[test]
    fn test_deform_conv_forward_dynamic_no_bias_no_mask() {
        let node =
            create_dynamic_deform_conv_node("deform_conv1", PaddingConfig2d::Valid, false, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            weight: Tensor<B, 4>,
            offset: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let output = burn::tensor::module::deform_conv2d(
                input,
                offset,
                weight,
                None,
                None,
                burn::tensor::ops::DeformConvOptions::new([1, 1], [0usize, 0usize], [1, 1], 1, 1),
            );
            output
        }
        ");
    }

    #[test]
    fn test_deform_conv_forward_dynamic_with_padding() {
        let node = create_dynamic_deform_conv_node(
            "deform_conv1",
            PaddingConfig2d::Explicit(1, 1, 1, 1),
            false,
            false,
        );
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            weight: Tensor<B, 4>,
            offset: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let output = burn::tensor::module::deform_conv2d(
                input,
                offset,
                weight,
                None,
                None,
                burn::tensor::ops::DeformConvOptions::new([1, 1], [1, 1], [1, 1], 1, 1),
            );
            output
        }
        ");
    }

    #[test]
    fn test_deform_conv_forward_dynamic_with_bias_and_mask() {
        let node =
            create_dynamic_deform_conv_node("deform_conv1", PaddingConfig2d::Valid, true, true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            weight: Tensor<B, 4>,
            offset: Tensor<B, 4>,
            bias: Tensor<B, 1>,
            mask: Tensor<B, 4>,
        ) -> Tensor<B, 4> {
            let output = burn::tensor::module::deform_conv2d(
                input,
                offset,
                weight,
                Some(mask),
                Some(bias),
                burn::tensor::ops::DeformConvOptions::new([1, 1], [0usize, 0usize], [1, 1], 1, 1),
            );
            output
        }
        ");
    }

    #[test]
    fn test_deform_conv_forward_dynamic_asymmetric_padding_emits_compile_error() {
        // Asymmetric padding (top != bottom) is not expressible via Burn's
        // symmetric `[usize; 2]` padding; codegen surfaces this as a
        // `compile_error!` token rather than panicking.
        let node = create_dynamic_deform_conv_node(
            "deform_conv1",
            PaddingConfig2d::Explicit(1, 1, 0, 1),
            false,
            false,
        );
        let code = codegen_forward_default(&node);
        assert!(
            code.contains("compile_error!")
                && code.contains("asymmetric padding (1, 1, 0, 1)")
                && code.contains("deform_conv1"),
            "expected compile_error! token naming the node and asymmetric values, got: {code}"
        );
    }
}
