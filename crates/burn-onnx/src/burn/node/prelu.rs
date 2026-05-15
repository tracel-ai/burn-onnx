use super::broadcast_helpers;
use super::prelude::*;
use burn::tensor::Shape;
use burn_store::TensorSnapshot;
use onnx_ir::ir::ArgType;
use onnx_ir::prelu::PReluNode;

/// True when the slope (input[1]) was lifted to a static initializer.
/// When false, slope arrives as a runtime forward parameter and we have to
/// implement PReLU element-wise instead of through `burn::nn::PRelu`.
fn slope_is_static(node: &PReluNode) -> bool {
    node.inputs.get(1).is_some_and(|s| s.is_static())
}

/// Calculate num_parameters from slope tensor's static shape.
fn num_parameters(node: &PReluNode) -> usize {
    node.inputs
        .get(1)
        .and_then(|slope| {
            if let ArgType::Tensor(tensor) = &slope.ty {
                tensor.static_shape_known()
            } else {
                None
            }
        })
        .map(|shape| shape.iter().product())
        .unwrap_or(1)
}

impl NodeCodegen for PReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !slope_is_static(self) {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let n = num_parameters(self).to_tokens();
        Some(Field::new(
            self.name.clone(),
            quote! { PRelu },
            quote! { let #name = PReluConfig::new().with_num_parameters(#n).init(device); },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        if let Some(alpha_input) = self.inputs.get(1) {
            let alpha_path = format!("{}.alpha", field_name);
            if let Some(mut snapshot) = create_lazy_snapshot(alpha_input, &alpha_path, "PRelu") {
                snapshot.shape = Shape::from([num_parameters(self)]);
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());
        let x = scope.arg(x_arg);

        if slope_is_static(self) {
            let field = Ident::new(&self.name, Span::call_site());
            return quote! {
                let #output = self.#field.forward(#x);
            };
        }

        // Right-align slope to x's rank so element-wise broadcasting follows the
        // ONNX numpy-style rules (slope shapes like `[C]` or `[C, 1, 1]` need
        // leading 1s prepended to match `x`'s rank before the multiply).
        let slope_arg = &self.inputs[1];
        let slope = scope.arg(slope_arg);
        let x_rank = x_arg.ty.rank();
        let slope_rank = slope_arg.ty.rank();
        let slope_aligned =
            broadcast_helpers::leading_broadcast(quote! { #slope }, slope_rank, x_rank);

        quote! {
            let #output = {
                let __x = #x;
                __x.clone()
                    .clamp_min(0_f64)
                    .add(#slope_aligned.mul(__x.clamp_max(0_f64)))
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if slope_is_static(self) {
            imports.register("burn::nn::PRelu");
            imports.register("burn::nn::PReluConfig");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::prelu::PReluNodeBuilder;

    #[test]
    fn test_prelu_forward_static_slope() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.prelu1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_prelu_field_with_channel_slope() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().with_num_parameters(64).init(device);");
    }

    #[test]
    fn test_prelu_field_with_scalar_slope() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("slope", vec![1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"let prelu1 = PReluConfig::new().with_num_parameters(1).init(device);");
    }

    #[test]
    fn test_prelu_forward_dynamic_slope_per_channel() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, slope: Tensor<3>) -> Tensor<4> {
            let output = {
                let __x = input;
                __x.clone()
                    .clamp_min(0_f64)
                    .add((slope).unsqueeze_dims(&[0isize]).mul(__x.clamp_max(0_f64)))
            };
            output
        }
        ");
    }

    #[test]
    fn test_prelu_forward_dynamic_slope_same_rank() {
        // Slope already matches x's rank: no leading_broadcast unsqueeze.
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("slope", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, slope: Tensor<3>) -> Tensor<3> {
            let output = {
                let __x = input;
                __x.clone().clamp_min(0_f64).add(slope.mul(__x.clamp_max(0_f64)))
            };
            output
        }
        ");
    }

    #[test]
    fn test_prelu_field_dynamic_slope_emits_no_field() {
        let node = PReluNodeBuilder::new("prelu1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("slope", vec![64, 1, 1], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build();
        let code = codegen_field_init(&node);
        assert_eq!(code, "");
    }
}
