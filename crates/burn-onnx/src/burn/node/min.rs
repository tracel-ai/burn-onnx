use super::prelude::*;

impl NodeCodegen for onnx_ir::node::min::MinNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs_arg = self.inputs.first().unwrap();
        let rhs_arg = self.inputs.get(1).unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());

        let lhs = scope.arg(lhs_arg);
        let rhs = scope.arg(rhs_arg);

        let function = match (&lhs_arg.ty, &rhs_arg.ty) {
            (lhs_ty, rhs_ty) if lhs_ty.is_on_device() && rhs_ty.is_on_device() => {
                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();

                if lhs_rank == rhs_rank {
                    quote! { #lhs.min_pair(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.min_pair(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).min_pair(#rhs) }
                }
            }
            _ => panic!("min: unsupported input types"),
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::min::{MinNode, MinNodeBuilder};

    fn create_min_node(name: &str, lhs_rank: usize, rhs_rank: usize) -> MinNode {
        MinNodeBuilder::new(name)
            .input_tensor("a", lhs_rank, DType::F32)
            .input_tensor("b", rhs_rank, DType::F32)
            .output_tensor("output", lhs_rank.max(rhs_rank), DType::F32)
            .build()
    }

    #[test]
    fn test_min() {
        let node = create_min_node("min1", 2, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = a.min_pair(b);
            output
        }
        ");
    }

    #[test]
    fn test_min_broadcast_lhs_larger() {
        let node = create_min_node("min1", 3, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3>, b: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = a.min_pair(b.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_min_broadcast_rhs_larger() {
        let node = create_min_node("min1", 2, 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2>, b: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = a.unsqueeze_dims(&[0isize]).min_pair(b);
            output
        }
        ");
    }
}
