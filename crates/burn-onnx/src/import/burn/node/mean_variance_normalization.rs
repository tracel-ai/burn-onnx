use super::prelude::*;
use onnx_ir::node::mean_variance_normalization::MeanVarianceNormalizationNode;

impl NodeCodegen for MeanVarianceNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // `mean_dims` folds over each axis with `mean_dim`, which keeps the
        // reduced dimension as size 1 so `mean` / `variance` broadcast back
        // against the original tensor shape.
        let axes: Vec<_> = self.config.axes.iter().map(|ax| ax.to_tokens()).collect();

        quote! {
            let #output = {
                let x = #input;
                let mean = x.clone().mean_dims(&[#(#axes),*]);
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32).mean_dims(&[#(#axes),*]);
                centered / variance.sqrt()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::mean_variance_normalization::{
        MeanVarianceNormalizationConfig, MeanVarianceNormalizationNodeBuilder,
    };

    #[test]
    fn default_axes_per_channel() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![0, 2, 3]))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let x = input;
                let mean = x.clone().mean_dims(&[0, 2, 3]);
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32).mean_dims(&[0, 2, 3]);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn single_axis() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![1]))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let x = input;
                let mean = x.clone().mean_dims(&[1]);
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32).mean_dims(&[1]);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }

    #[test]
    fn forward_with_clone() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(MeanVarianceNormalizationConfig::new(vec![0, 2, 3]))
            .build();
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = {
                let x = input.clone();
                let mean = x.clone().mean_dims(&[0, 2, 3]);
                let centered = x - mean;
                let variance = centered.clone().powf_scalar(2f32).mean_dims(&[0, 2, 3]);
                centered / variance.sqrt()
            };
            output
        }
        ");
    }
}
