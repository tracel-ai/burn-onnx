use super::prelude::*;
impl NodeCodegen for onnx_ir::dynamic_quantize_linear::DynamicQuantizeLinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }
    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }
    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let input = if matches!(self.inputs[0].ty, ArgType::ScalarNative(_)) {
            quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([#input]), (self.device(), burn::tensor::DType::F32)) }
        } else {
            input
        };
        let y = arg_to_ident(&self.outputs[0]);
        let scale = arg_to_ident(&self.outputs[1]);
        let zero = arg_to_ident(&self.outputs[2]);
        let rank = self.inputs[0].ty.rank().max(1);
        quote! {
            let (#y, #scale, #zero) = {
                let input = #input;
                let min = input.clone().min().clamp_max(0.0);
                let max = input.clone().max().clamp_min(0.0);
                let range = max - min.clone();
                let scale = range.clone().mask_fill(range.equal_elem(0.0), 1.0).div_scalar(255.0);
                let zero = (min.neg() / scale.clone()).clamp(0.0, 255.0).round();
                let y = (input / scale.clone().reshape([1; #rank])).round()
                    .add(zero.clone().reshape([1; #rank])).clamp(0.0, 255.0)
                    .int().cast(burn::tensor::DType::U8);
                (y, scale, zero.int().cast(burn::tensor::DType::U8))
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::dynamic_quantize_linear::DynamicQuantizeLinearNodeBuilder;
    #[test]
    fn dynamic_quantization() {
        let node = DynamicQuantizeLinearNodeBuilder::new("dynamic")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("y", 2, DType::U8)
            .output_scalar_tensor("scale", DType::F32)
            .output_scalar_tensor("zero", DType::U8)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: Tensor<2>) -> (Tensor<2, Int>, Tensor<1>, Tensor<1, Int>) {
            let (y, scale, zero) = {
                let input = input;
                let min = input.clone().min().clamp_max(0.0);
                let max = input.clone().max().clamp_min(0.0);
                let range = max - min.clone();
                let scale = range
                    .clone()
                    .mask_fill(range.equal_elem(0.0), 1.0)
                    .div_scalar(255.0);
                let zero = (min.neg() / scale.clone()).clamp(0.0, 255.0).round();
                let y = (input / scale.clone().reshape([1; 2usize]))
                    .round()
                    .add(zero.clone().reshape([1; 2usize]))
                    .clamp(0.0, 255.0)
                    .int()
                    .cast(burn::tensor::DType::U8);
                (y, scale, zero.int().cast(burn::tensor::DType::U8))
            };
            (y, scale, zero)
        }
        ");
    }
    #[test]
    fn native_scalar_input() {
        let node = DynamicQuantizeLinearNodeBuilder::new("dynamic")
            .input_scalar("input", DType::F32)
            .output_scalar_tensor("y", DType::U8)
            .output_scalar_tensor("scale", DType::F32)
            .output_scalar_tensor("zero", DType::U8)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, input: f32) -> (Tensor<1, Int>, Tensor<1>, Tensor<1, Int>) {
            let (y, scale, zero) = {
                let input = burn::tensor::Tensor::<
                    1,
                >::from_data(
                    burn::tensor::TensorData::from([input]),
                    (self.device(), burn::tensor::DType::F32),
                );
                let min = input.clone().min().clamp_max(0.0);
                let max = input.clone().max().clamp_min(0.0);
                let range = max - min.clone();
                let scale = range
                    .clone()
                    .mask_fill(range.equal_elem(0.0), 1.0)
                    .div_scalar(255.0);
                let zero = (min.neg() / scale.clone()).clamp(0.0, 255.0).round();
                let y = (input / scale.clone().reshape([1; 1usize]))
                    .round()
                    .add(zero.clone().reshape([1; 1usize]))
                    .clamp(0.0, 255.0)
                    .int()
                    .cast(burn::tensor::DType::U8);
                (y, scale, zero.int().cast(burn::tensor::DType::U8))
            };
            (y, scale, zero)
        }
        ");
    }
}
