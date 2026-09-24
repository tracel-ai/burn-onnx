use super::prelude::*;

impl NodeCodegen for onnx_ir::global_lp_pool::GlobalLpPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let p = self.config.p;

        // onnx-ir rejects each of the conditions below, so they are reachable only from
        // a hand-built node: the config and the node builder are public and don't
        // validate. Emitting a named `compile_error!` keeps the failure inside the
        // generated crate instead of crashing model generation.
        let invalid = |reason: String| {
            let msg = format!("GlobalLpPool node '{}': {reason}", self.name);
            quote! { let #output = { compile_error!(#msg); unreachable!() }; }
        };

        let rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            ty => return invalid(format!("input must be a tensor, got {ty:?}")),
        };
        // Below rank 3 there are no spatial axes, and `lp_norm_dims` over an empty slice
        // reduces nothing, silently applying |x| elementwise instead.
        if rank <= 2 {
            return invalid(format!("requires rank >= 3, got rank {rank}"));
        }
        // `is_finite` also rules out NaN, and a non-finite literal would panic inside
        // proc-macro2. `lp_norm_dims` reads p = 0 as the L0 count, which is not a pool.
        if !p.is_finite() || p <= 0.0 {
            return invalid(format!("p must be finite and > 0, got {p}"));
        }

        // N and C carry through; every spatial axis reduces to size 1, giving the
        // [N, C, 1, 1, ...] output the spec requires.
        let dims = (2..rank).collect::<Vec<usize>>().to_tokens();

        // `lp_norm_dims` computes in the tensor's own dtype, so a large p on an f16
        // input can overflow (f16 saturates at 65504); that matches ORT, which also
        // evaluates in the input dtype.
        quote! {
            let #output = burn::tensor::linalg::lp_norm_dims(#input, #p, &#dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_lp_pool::{GlobalLpPoolConfig, GlobalLpPoolNodeBuilder};

    fn code_for(rank: usize, p: f64) -> String {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_tensor("input", rank, DType::F32)
            .output_tensor("output", rank, DType::F32)
            .config(GlobalLpPoolConfig::new(p))
            .build();
        codegen_forward_default(&node)
    }

    #[test]
    fn global_lp_pool_rank3_l1() {
        assert_snapshot!(code_for(3, 1.0), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 1f64, &[2]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l2() {
        assert_snapshot!(code_for(3, 2.0), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 2f64, &[2]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank3_l3() {
        assert_snapshot!(code_for(3, 3.0), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 3f64, &[2]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l1() {
        assert_snapshot!(code_for(5, 1.0), @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 1f64, &[2, 3, 4]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank5_l2() {
        assert_snapshot!(code_for(5, 2.0), @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 2f64, &[2, 3, 4]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank6_l8() {
        assert_snapshot!(code_for(6, 8.0), @r"
        pub fn forward(&self, input: Tensor<6>) -> Tensor<6> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 8f64, &[2, 3, 4, 5]);
            output
        }
        ");
    }

    /// Opset 1 allows a fractional p, which takes `powf_scalar` and always needs the
    /// `abs` (a negative base with a fractional exponent is NaN).
    #[test]
    fn global_lp_pool_rank3_fractional_p() {
        assert_snapshot!(code_for(3, 2.5), @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = burn::tensor::linalg::lp_norm_dims(input, 2.5f64, &[2]);
            output
        }
        ");
    }

    #[test]
    fn global_lp_pool_rank2_emits_compile_error() {
        assert_snapshot!(code_for(2, 2.0), @r#"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = {
                compile_error!(
                    "GlobalLpPool node 'global_lp_pool': requires rank >= 3, got rank 2"
                );
                unreachable!()
            };
            output
        }
        "#);
    }

    #[test]
    fn global_lp_pool_non_positive_p_emits_compile_error() {
        assert_snapshot!(code_for(3, 0.0), @r#"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                compile_error!(
                    "GlobalLpPool node 'global_lp_pool': p must be finite and > 0, got 0"
                );
                unreachable!()
            };
            output
        }
        "#);
    }

    #[test]
    fn global_lp_pool_non_tensor_input_emits_compile_error() {
        let node = GlobalLpPoolNodeBuilder::new("global_lp_pool")
            .input_scalar("input", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(GlobalLpPoolConfig::new(2.0))
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, input: f32) -> Tensor<3> {
            let output = {
                compile_error!(
                    "GlobalLpPool node 'global_lp_pool': input must be a tensor, got ScalarNative(F32)"
                );
                unreachable!()
            };
            output
        }
        "#);
    }
}
