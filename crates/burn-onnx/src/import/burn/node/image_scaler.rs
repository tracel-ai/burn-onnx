use super::prelude::*;
impl NodeCodegen for onnx_ir::image_scaler::ImageScalerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }
    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }
    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);
        let scale = super::super::codegen::f32_to_tokens(self.config.scale);
        if self.config.bias.is_empty() {
            return quote! { let #output = #input.mul_scalar(#scale); };
        }
        let channels = self.config.bias.len();
        let bias = self
            .config
            .bias
            .iter()
            .map(|v| super::super::codegen::f32_to_tokens(*v));
        let dtype = self.inputs[0].ty.elem_type().to_tokens();
        quote! {
            let #output = {
                let scaled = #input.mul_scalar(#scale);
                assert_eq!(scaled.dims()[1], #channels, "ImageScaler bias must match the channel count");
                let bias = burn::tensor::Tensor::<4>::from_data(
                    burn::tensor::TensorData::new(alloc::vec![#(#bias),*], [1, #channels, 1, 1]),
                    (&scaled.device(), #dtype),
                );
                scaled + bias
            };
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::image_scaler::{ImageScalerConfig, ImageScalerNodeBuilder};
    #[test]
    fn default_scale() {
        let node = ImageScalerNodeBuilder::new("scale")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(ImageScalerConfig::default())
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = input.mul_scalar(1f32);
            output
        }
        ");
    }
    #[test]
    fn channel_bias() {
        let node = ImageScalerNodeBuilder::new("scale")
            .input_tensor("bias", 4, DType::F64)
            .output_tensor("output", 4, DType::F64)
            .config(ImageScalerConfig {
                scale: 0.5,
                bias: vec![1.0, -2.0, 3.0],
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, bias: Tensor<4>) -> Tensor<4> {
            let output = {
                let scaled = bias.mul_scalar(0.5f32);
                assert_eq!(
                    scaled.dims() [1], 3usize, "ImageScaler bias must match the channel count"
                );
                let bias = burn::tensor::Tensor::<
                    4,
                >::from_data(
                    burn::tensor::TensorData::new(
                        alloc::vec![1f32, - 2f32, 3f32],
                        [1, 3usize, 1, 1],
                    ),
                    (&scaled.device(), burn::tensor::DType::F64),
                );
                scaled + bias
            };
            output
        }
        "#);
    }
    #[test]
    fn non_finite_attributes() {
        let node = ImageScalerNodeBuilder::new("scale")
            .input_tensor("input", 4, DType::F64)
            .output_tensor("output", 4, DType::F64)
            .config(ImageScalerConfig {
                scale: f32::INFINITY,
                bias: vec![f32::NAN, f32::NEG_INFINITY, 0.1],
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let scaled = input.mul_scalar(f32::INFINITY);
                assert_eq!(
                    scaled.dims() [1], 3usize, "ImageScaler bias must match the channel count"
                );
                let bias = burn::tensor::Tensor::<
                    4,
                >::from_data(
                    burn::tensor::TensorData::new(
                        alloc::vec![f32::NAN, f32::NEG_INFINITY, 0.1f32],
                        [1, 3usize, 1, 1],
                    ),
                    (&scaled.device(), burn::tensor::DType::F64),
                );
                scaled + bias
            };
            output
        }
        "#);
    }
}
