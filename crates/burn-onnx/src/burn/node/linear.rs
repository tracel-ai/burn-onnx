use super::prelude::*;
use burn_store::TensorSnapshot;

impl NodeCodegen for onnx_ir::linear::LinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let weight_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("Linear: weight tensor shape must be known at codegen time");
        let (d_input, d_output) = if self.config.transpose_weight {
            // Gemm layout: [out_features, in_features]
            (weight_shape[1].to_tokens(), weight_shape[0].to_tokens())
        } else {
            // MatMul layout: [in_features, out_features]
            (weight_shape[0].to_tokens(), weight_shape[1].to_tokens())
        };
        let bias = self.inputs.len() > 2;

        // ONNX Gemm stores weights as [d_output, d_input], which matches LinearLayout::Col.
        // MatMul-sourced Linear stores weights as [d_input, d_output], matching LinearLayout::Row.
        // Using the appropriate layout avoids data transposition during import.
        let init_code = if self.config.transpose_weight {
            quote! {
                let #name = LinearConfig::new(#d_input, #d_output)
                    .with_bias(#bias)
                    .with_layout(LinearLayout::Col)
                    .init(device);
            }
        } else {
            quote! {
                let #name = LinearConfig::new(#d_input, #d_output)
                    .with_bias(#bias)
                    .init(device);
            }
        };

        Some(Field::new(
            self.name.clone(),
            quote! { Linear<B> },
            init_code,
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        // Weight tensor (input index 1)
        // No transposition needed - LinearLayout::Col handles ONNX [out, in] format
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(snapshot) = create_lazy_snapshot(weight_input, &weight_path, "Linear") {
                snapshots.push(snapshot);
            }
        }

        // Bias tensor (input index 2, optional)
        if let Some(bias_input) = self.inputs.get(2) {
            let bias_path = format!("{}.bias", field_name);
            if let Some(snapshot) = create_lazy_snapshot(bias_input, &bias_path, "Linear") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::Linear");
        imports.register("burn::nn::LinearConfig");
        if self.config.transpose_weight {
            imports.register("burn::nn::LinearLayout");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType, ValueSource};
    use onnx_ir::linear::{LinearConfig, LinearNode};

    fn static_tensor_arg(name: &str, shape: Vec<usize>, dtype: DType) -> Argument {
        let mut arg = Argument::new(name, ArgType::Tensor(TensorType::new_known(dtype, shape)));
        arg.value_source = ValueSource::Static(0);
        arg
    }

    fn create_linear_node_gemm(name: &str) -> LinearNode {
        // Gemm-sourced: transpose_weight=true, weight is [out=64, in=128]
        let config = LinearConfig::new(true);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = static_tensor_arg("weight", vec![64, 128], DType::F32);
        let bias = static_tensor_arg("bias", vec![64], DType::F32);

        LinearNode {
            name: name.to_string(),
            inputs: vec![input, weight, bias],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        }
    }

    fn create_linear_node_matmul(name: &str) -> LinearNode {
        // MatMul-sourced: transpose_weight=false, weight is [in=128, out=64]
        let config = LinearConfig::new(false);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = static_tensor_arg("weight", vec![128, 64], DType::F32);

        LinearNode {
            name: name.to_string(),
            inputs: vec![input, weight],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        }
    }

    #[test]
    fn test_linear_forward() {
        let node = create_linear_node_gemm("linear1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = self.linear1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let node = create_linear_node_matmul("linear2");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = self.linear2.forward(input);
            output
        }
        ");
    }
}
