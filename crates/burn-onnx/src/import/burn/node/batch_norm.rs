use super::prelude::*;
use burn_pack::Tensor as PackTensor;
use onnx_ir::node::batch_norm::{BatchNormConfig, BatchNormalizationNode};

impl NodeCodegen for BatchNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        match &self.config {
            BatchNormConfig::Static(config) => {
                let name = Ident::new(&self.name, Span::call_site());
                let scale_shape = self.inputs[1]
                    .ty
                    .static_shape_known()
                    .expect("BatchNorm: scale tensor shape must be known at codegen time");
                let num_features = scale_shape[0].to_tokens();
                let epsilon = config.epsilon;
                let momentum = config.momentum;

                Some(Field::new(
                    self.name.clone(),
                    quote! {
                        BatchNorm
                    },
                    quote! {
                        let #name = BatchNormConfig::new(#num_features)
                            .with_epsilon(#epsilon)
                            .with_momentum(#momentum)
                            .init(device);
                    },
                ))
            }
            BatchNormConfig::Runtime(_) => None,
        }
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        match &self.config {
            BatchNormConfig::Static(_) => {
                let input = scope.arg(self.inputs.first().unwrap());
                let field = Ident::new(&self.name, Span::call_site());

                quote! {
                    let #output = self.#field.forward(#input);
                }
            }
            BatchNormConfig::Runtime(config) => {
                let input = scope.arg(&self.inputs[0]);
                let scale = scope.arg(&self.inputs[1]);
                let bias = scope.arg(&self.inputs[2]);
                let mean = scope.arg(&self.inputs[3]);
                let var = scope.arg(&self.inputs[4]);
                let epsilon = config.epsilon;

                quote! {
                    let #output = burn::tensor::module::batch_norm(
                        #input, #scale, #bias, #mean, #var, #epsilon,
                    );
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.config {
            BatchNormConfig::Static(_) => {
                imports.register("burn::nn::BatchNorm");
                imports.register("burn::nn::BatchNormConfig");
            }
            BatchNormConfig::Runtime(_) => {}
        }
    }

    fn collect_tensors(&self, field_name: &str) -> Vec<PackTensor> {
        match &self.config {
            BatchNormConfig::Static(_) => {
                use crate::burn::node_traits::create_deferred_tensor;
                let mut tensors = vec![];

                if let Some(gamma_input) = self.inputs.get(1) {
                    let gamma_path = format!("{}.gamma", field_name);
                    if let Some(tensor) = create_deferred_tensor(gamma_input, &gamma_path) {
                        tensors.push(tensor);
                    }
                }

                if let Some(beta_input) = self.inputs.get(2) {
                    let beta_path = format!("{}.beta", field_name);
                    if let Some(tensor) = create_deferred_tensor(beta_input, &beta_path) {
                        tensors.push(tensor);
                    }
                }

                if let Some(running_mean_input) = self.inputs.get(3) {
                    let running_mean_path = format!("{}.running_mean", field_name);
                    if let Some(tensor) =
                        create_deferred_tensor(running_mean_input, &running_mean_path)
                    {
                        tensors.push(tensor);
                    }
                }

                if let Some(running_var_input) = self.inputs.get(4) {
                    let running_var_path = format!("{}.running_var", field_name);
                    if let Some(tensor) =
                        create_deferred_tensor(running_var_input, &running_var_path)
                    {
                        tensors.push(tensor);
                    }
                }

                tensors
            }
            BatchNormConfig::Runtime(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::batch_norm::{
        BatchNormConfig, BatchNormRuntimeConfig, BatchNormStaticConfig, BatchNormalizationNode,
        BatchNormalizationNodeBuilder,
    };

    fn create_batch_norm_node(name: &str) -> BatchNormalizationNode {
        let config = BatchNormConfig::Static(BatchNormStaticConfig::new(1e-5, 0.9));

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("scale", vec![64], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .input_static_tensor_shape("mean", vec![64], DType::F32)
            .input_static_tensor_shape("var", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_runtime_batch_norm_node(name: &str, input_rank: usize) -> BatchNormalizationNode {
        let config = BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(1e-5, 0.9));

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", input_rank, DType::F32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .input_tensor("mean", 1, DType::F32)
            .input_tensor("var", 1, DType::F32)
            .output_tensor("output", input_rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_batch_norm_forward() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.batch_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_forward_with_clone() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.batch_norm1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank3() {
        let node = create_runtime_batch_norm_node("batch_norm1", 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            scale: Tensor<1>,
            bias: Tensor<1>,
            mean: Tensor<1>,
            var: Tensor<1>,
        ) -> Tensor<3> {
            let output = burn::tensor::module::batch_norm(
                input,
                scale,
                bias,
                mean,
                var,
                0.00001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank4() {
        let node = create_runtime_batch_norm_node("batch_norm1", 4);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<4>,
            scale: Tensor<1>,
            bias: Tensor<1>,
            mean: Tensor<1>,
            var: Tensor<1>,
        ) -> Tensor<4> {
            let output = burn::tensor::module::batch_norm(
                input,
                scale,
                bias,
                mean,
                var,
                0.00001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank5() {
        let node = create_runtime_batch_norm_node("batch_norm1", 5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<5>,
            scale: Tensor<1>,
            bias: Tensor<1>,
            mean: Tensor<1>,
            var: Tensor<1>,
        ) -> Tensor<5> {
            let output = burn::tensor::module::batch_norm(
                input,
                scale,
                bias,
                mean,
                var,
                0.00001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_with_clone() {
        let node = create_runtime_batch_norm_node("batch_norm1", 4);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<4>,
            scale: Tensor<1>,
            bias: Tensor<1>,
            mean: Tensor<1>,
            var: Tensor<1>,
        ) -> Tensor<4> {
            let output = burn::tensor::module::batch_norm(
                input.clone(),
                scale.clone(),
                bias.clone(),
                mean.clone(),
                var.clone(),
                0.00001f64,
            );
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank2() {
        let node = create_runtime_batch_norm_node("batch_norm1", 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<2>,
            scale: Tensor<1>,
            bias: Tensor<1>,
            mean: Tensor<1>,
            var: Tensor<1>,
        ) -> Tensor<2> {
            let output = burn::tensor::module::batch_norm(
                input,
                scale,
                bias,
                mean,
                var,
                0.00001f64,
            );
            output
        }
        ");
    }
}
