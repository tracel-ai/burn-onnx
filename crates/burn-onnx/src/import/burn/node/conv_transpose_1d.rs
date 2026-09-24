use super::prelude::*;
use burn_pack::Tensor as PackTensor;

impl NodeCodegen for onnx_ir::node::conv_transpose1d::ConvTranspose1dNode {
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
            .expect("ConvTranspose1d: weight tensor shape must be known at codegen time");
        let groups = self.config.groups;
        let channels_in = weight_shape[0].to_tokens();
        let channels_out = (weight_shape[1] * groups).to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let padding_out = self.config.padding_out.to_tokens();
        let bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

        Some(Field::new(
            self.name.clone(),
            quote! {
                ConvTranspose1d
            },
            quote! {
                let #name = ConvTranspose1dConfig::new([#channels_in, #channels_out], #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_padding_out(#padding_out)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // A runtime weight has no module to live in, so the functional op takes it.
        if !self.inputs[1].is_static() {
            let weight = scope.arg(&self.inputs[1]);
            let bias = super::conv_helpers::optional_input(scope, self.inputs.get(2));
            let stride = [self.config.stride].to_tokens();
            let padding = [self.config.padding].to_tokens();
            let padding_out = [self.config.padding_out].to_tokens();
            let dilation = [self.config.dilation].to_tokens();
            let groups = self.config.groups.to_tokens();
            return quote! {
                let #output = burn::tensor::module::conv_transpose1d(
                    #input,
                    #weight,
                    #bias,
                    burn::tensor::ops::ConvTransposeOptions::new(
                        #stride,
                        #padding,
                        #padding_out,
                        #dilation,
                        #groups,
                    ),
                );
            };
        }
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }
    fn register_imports(&self, imports: &mut BurnImports) {
        if !self.inputs[1].is_static() {
            return;
        }
        imports.register("burn::nn::conv::ConvTranspose1d");
        imports.register("burn::nn::conv::ConvTranspose1dConfig");
    }

    fn collect_tensors(&self, field_name: &str) -> Vec<PackTensor> {
        if !self.inputs[1].is_static() {
            return vec![];
        }
        use crate::burn::node_traits::create_deferred_tensor;

        let mut tensors = vec![];

        // Weight tensor (input index 1)
        // ONNX ConvTranspose weight: [in_channels, out_channels/groups, kL]
        // Burn ConvTranspose1d weight: [channels_in, channels_out/groups, kernel_size]
        // These layouts match! No transformation needed.
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
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::conv_transpose1d::{
        ConvTranspose1dConfig, ConvTranspose1dNode, ConvTranspose1dNodeBuilder,
    };

    fn create_conv_transpose_1d_node(name: &str) -> ConvTranspose1dNode {
        let config = ConvTranspose1dConfig::new(3, 1, 1, 1, 1, 0);

        ConvTranspose1dNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .input_static_tensor_shape("weight", vec![3, 64, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv_transpose_1d_forward() {
        let node = create_conv_transpose_1d_node("conv_transpose1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.conv_transpose1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv_transpose_1d_forward_with_clone() {
        let node = create_conv_transpose_1d_node("conv_transpose1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.conv_transpose1.forward(input.clone());
            output
        }
        ");
    }
    #[test]
    fn test_conv_transpose_1d_runtime_weight() {
        let node = {
            let config = ConvTranspose1dConfig::new(3, 1, 1, 1, 1, 0);

            ConvTranspose1dNodeBuilder::new("conv1")
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
            let output = burn::tensor::module::conv_transpose1d(
                input,
                weight,
                Some(bias),
                burn::tensor::ops::ConvTransposeOptions::new([1], [1], [0], [1], 1),
            );
            output
        }
        ");
    }
}
