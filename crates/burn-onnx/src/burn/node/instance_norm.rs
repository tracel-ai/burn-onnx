use super::broadcast_helpers::channel_broadcast_shape;
use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::node::instance_norm::InstanceNormalizationNode;

/// True when both scale (input[1]) and bias (input[2]) were lifted to static
/// initializers. When either is dynamic, the Burn `InstanceNorm` module field
/// can't be populated from the burnpack, so we inline the formula instead.
fn weights_are_static(node: &InstanceNormalizationNode) -> bool {
    node.inputs.get(1).is_some_and(|s| s.is_static())
        && node.inputs.get(2).is_some_and(|b| b.is_static())
}

impl NodeCodegen for InstanceNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !weights_are_static(self) {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let scale_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("InstanceNorm: scale tensor shape must be known at codegen time");
        let num_features = scale_shape[0].to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! { InstanceNorm },
            quote! {
                let #name = InstanceNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        if let Some(gamma_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(snapshot) = create_lazy_snapshot(gamma_input, &gamma_path, "InstanceNorm") {
                snapshots.push(snapshot);
            }
        }

        if let Some(beta_input) = self.inputs.get(2) {
            let beta_path = format!("{}.beta", field_name);
            if let Some(snapshot) = create_lazy_snapshot(beta_input, &beta_path, "InstanceNorm") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());
        let x = scope.arg(x_arg);

        if weights_are_static(self) {
            let field = Ident::new(&self.name, Span::call_site());
            return quote! {
                let #output = self.#field.forward(#x);
            };
        }

        // Spatial dims are collapsed into a single hidden axis before reducing
        // (mirrors Burn's internal `group_norm` shape strategy). The channel
        // dim for the affine reshape is read from `__dims` at runtime so we
        // don't need scale's static shape to be known at codegen time.
        let scale = scope.arg(&self.inputs[1]);
        let bias = scope.arg(&self.inputs[2]);
        let rank = x_arg.ty.rank();
        let epsilon = self.config.epsilon;
        let affine_shape = channel_broadcast_shape(rank, quote! { __channels });

        quote! {
            let #output = {
                let __x = #x;
                let __dims = __x.dims();
                let __batch = __dims[0];
                let __channels = __dims[1];
                let __hidden: usize = __dims[2..].iter().product();
                let __hidden_f = __hidden as f64;
                let __x3 = __x.reshape([__batch, __channels, __hidden]);
                let __mean = __x3.clone().sum_dim(2).div_scalar(__hidden_f);
                let __centered = __x3.sub(__mean);
                let __var = __centered.clone().square().sum_dim(2).div_scalar(__hidden_f);
                let __normalized = __centered.div(__var.add_scalar(#epsilon).sqrt());
                __normalized
                    .reshape(__dims)
                    .mul(#scale.reshape(#affine_shape))
                    .add(#bias.reshape(#affine_shape))
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if weights_are_static(self) {
            imports.register("burn::nn::InstanceNorm");
            imports.register("burn::nn::InstanceNormConfig");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::instance_norm::{
        InstanceNormConfig, InstanceNormalizationNode, InstanceNormalizationNodeBuilder,
    };

    fn create_static_instance_norm_node(name: &str) -> InstanceNormalizationNode {
        let config = InstanceNormConfig::new(1e-5);
        InstanceNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("scale", vec![32], DType::F32)
            .input_static_tensor_shape("bias", vec![32], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_dynamic_instance_norm_node(name: &str) -> InstanceNormalizationNode {
        let config = InstanceNormConfig::new(1e-5);
        InstanceNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("scale", vec![2], DType::F32)
            .input_tensor_shape("bias", vec![2], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_instance_norm_forward_static() {
        let node = create_static_instance_norm_node("instance_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.instance_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_instance_norm_forward_static_with_clone() {
        let node = create_static_instance_norm_node("instance_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.instance_norm1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_instance_norm_forward_dynamic() {
        let node = create_dynamic_instance_norm_node("instance_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, scale: Tensor<1>, bias: Tensor<1>) -> Tensor<4> {
            let output = {
                let __x = input;
                let __dims = __x.dims();
                let __batch = __dims[0];
                let __channels = __dims[1];
                let __hidden: usize = __dims[2..].iter().product();
                let __hidden_f = __hidden as f64;
                let __x3 = __x.reshape([__batch, __channels, __hidden]);
                let __mean = __x3.clone().sum_dim(2).div_scalar(__hidden_f);
                let __centered = __x3.sub(__mean);
                let __var = __centered.clone().square().sum_dim(2).div_scalar(__hidden_f);
                let __normalized = __centered.div(__var.add_scalar(0.00001f64).sqrt());
                __normalized
                    .reshape(__dims)
                    .mul(scale.reshape([1usize, __channels, 1usize, 1usize]))
                    .add(bias.reshape([1usize, __channels, 1usize, 1usize]))
            };
            output
        }
        ");
    }

    #[test]
    fn test_instance_norm_field_dynamic_emits_no_field() {
        let node = create_dynamic_instance_norm_node("instance_norm1");
        let code = codegen_field_init(&node);
        assert_eq!(code, "");
    }
}
