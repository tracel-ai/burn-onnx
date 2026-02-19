use super::prelude::*;

/// Type of power operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerType {
    /// Integer power (powi)
    Int,
    /// Float power (powf)
    Float,
}

impl NodeCodegen for onnx_ir::pow::PowNode {
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

        // Determine power type based on RHS type
        let power_type = match &rhs_arg.ty {
            ArgType::Tensor(t) => match &t.dtype {
                dtype if dtype.is_int() => PowerType::Int,
                dtype if dtype.is_float() => PowerType::Float,
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            ArgType::ScalarTensor(dtype) => match dtype {
                dtype if dtype.is_int() => PowerType::Int,
                dtype if dtype.is_float() => PowerType::Float,
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            ArgType::ScalarNative(dtype) => match dtype {
                dtype if dtype.is_int() => PowerType::Int,
                dtype if dtype.is_float() => PowerType::Float,
                _ => panic!("pow function requires RHS to be int or float type"),
            },
            _ => panic!("pow function only supports RHS scalar or tensor types"),
        };

        let function = match (power_type, &lhs_arg.ty, &rhs_arg.ty) {
            (PowerType::Int, lhs_ty, rhs_ty) if lhs_ty.is_on_device() && rhs_ty.is_on_device() => {
                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();
                if lhs_rank == rhs_rank {
                    quote! { #lhs.powi(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.powi(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).powi(#rhs) }
                }
            }
            (PowerType::Float, lhs_ty, rhs_ty)
                if lhs_ty.is_on_device() && rhs_ty.is_on_device() =>
            {
                let lhs_rank = lhs_ty.rank();
                let rhs_rank = rhs_ty.rank();
                if lhs_rank == rhs_rank {
                    quote! { #lhs.powf(#rhs) }
                } else if lhs_rank > rhs_rank {
                    let num_dims = lhs_rank - rhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.powf(#rhs.unsqueeze_dims(&[#(#dims),*])) }
                } else {
                    let num_dims = rhs_rank - lhs_rank;
                    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                    quote! { #lhs.unsqueeze_dims(&[#(#dims),*]).powf(#rhs) }
                }
            }
            // ScalarNative + ScalarNative (native Rust pow)
            (PowerType::Float, ArgType::ScalarNative(_), ArgType::ScalarNative(_)) => {
                quote! { #lhs.powf(#rhs) }
            }
            (PowerType::Int, ArgType::ScalarNative(lhs_dtype), ArgType::ScalarNative(_)) => {
                if lhs_dtype.is_float() {
                    quote! { #lhs.powi(#rhs as i32) }
                } else {
                    quote! { #lhs.pow(#rhs as u32) }
                }
            }
            // ScalarNative + on_device (promote scalar to tensor)
            (PowerType::Float, ArgType::ScalarNative(dtype), rhs_ty) if rhs_ty.is_on_device() => {
                let dtype_tokens = dtype.to_tokens();
                let rhs_rank = rhs_ty.rank();
                let base = if rhs_rank > 1 {
                    let dims: Vec<isize> = (0..rhs_rank - 1).map(|i| i as isize).collect();
                    quote! {
                        Tensor::<B, 1>::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as f64]),
                            &*self.device,
                            #dtype_tokens
                        ).unsqueeze_dims(&[#(#dims),*])
                    }
                } else {
                    quote! {
                        Tensor::<B, 1>::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as f64]),
                            &*self.device,
                            #dtype_tokens
                        )
                    }
                };
                quote! { #base.powf(#rhs) }
            }
            (PowerType::Int, ArgType::ScalarNative(dtype), rhs_ty)
                if rhs_ty.is_on_device() && dtype.is_float() =>
            {
                let dtype_tokens = dtype.to_tokens();
                let rhs_rank = rhs_ty.rank();
                let base = if rhs_rank > 1 {
                    let dims: Vec<isize> = (0..rhs_rank - 1).map(|i| i as isize).collect();
                    quote! {
                        Tensor::<B, 1>::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as f64]),
                            &*self.device,
                            #dtype_tokens
                        ).unsqueeze_dims(&[#(#dims),*])
                    }
                } else {
                    quote! {
                        Tensor::<B, 1>::from_data_dtype(
                            burn::tensor::TensorData::from([#lhs as f64]),
                            &*self.device,
                            #dtype_tokens
                        )
                    }
                };
                quote! { #base.powi(#rhs) }
            }
            // on_device + ScalarNative
            (PowerType::Int, _, ArgType::ScalarNative(_)) => quote! { #lhs.powi_scalar(#rhs) },
            (PowerType::Float, _, ArgType::ScalarNative(_)) => quote! { #lhs.powf_scalar(#rhs) },
            (pt, lhs, rhs) => panic!(
                "Unsupported pow type combination: power_type={pt:?}, lhs={lhs:?}, rhs={rhs:?}"
            ),
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
    use onnx_ir::pow::PowNodeBuilder;

    // --- on_device + on_device (float power) ---

    #[test]
    fn test_powf_tensor_tensor_same_rank() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_tensor("exponent", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = base.powf(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powf_broadcast_lhs_higher() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 3, DType::F32)
            .input_tensor("exponent", 2, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 3>, exponent: Tensor<B, 2>) -> Tensor<B, 3> {
            let output = base.powf(exponent.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_powf_broadcast_rhs_higher() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_tensor("exponent", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = base.unsqueeze_dims(&[0isize]).powf(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powf_tensor_scalar_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 3, DType::F32)
            .input_scalar_tensor("exponent", DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 3>, exponent: Tensor<B, 1>) -> Tensor<B, 3> {
            let output = base.powf(exponent.unsqueeze_dims(&[0isize, 1isize]));
            output
        }
        ");
    }

    #[test]
    fn test_powf_scalar_tensor_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar_tensor("base", DType::F32)
            .input_tensor("exponent", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 1>, exponent: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = base.unsqueeze_dims(&[0isize, 1isize]).powf(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powf_scalar_tensor_scalar_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar_tensor("base", DType::F32)
            .input_scalar_tensor("exponent", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 1>, exponent: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = base.powf(exponent);
            output
        }
        ");
    }

    // --- on_device + on_device (int power) ---

    #[test]
    fn test_powi_tensor_tensor_same_rank() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_tensor("exponent", 2, DType::I32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = base.powi(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_broadcast_lhs_higher() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 3, DType::F32)
            .input_tensor("exponent", 2, DType::I32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 3>, exponent: Tensor<B, 2, Int>) -> Tensor<B, 3> {
            let output = base.powi(exponent.unsqueeze_dims(&[0isize]));
            output
        }
        ");
    }

    #[test]
    fn test_powi_broadcast_rhs_higher() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_tensor("exponent", 3, DType::I32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: Tensor<B, 3, Int>) -> Tensor<B, 3> {
            let output = base.unsqueeze_dims(&[0isize]).powi(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_tensor_scalar_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 3, DType::F32)
            .input_scalar_tensor("exponent", DType::I32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 3>, exponent: Tensor<B, 1, Int>) -> Tensor<B, 3> {
            let output = base.powi(exponent.unsqueeze_dims(&[0isize, 1isize]));
            output
        }
        ");
    }

    #[test]
    fn test_powi_scalar_tensor_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar_tensor("base", DType::F32)
            .input_tensor("exponent", 3, DType::I32)
            .output_tensor("output", 3, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 1>, exponent: Tensor<B, 3, Int>) -> Tensor<B, 3> {
            let output = base.unsqueeze_dims(&[0isize, 1isize]).powi(exponent);
            output
        }
        ");
    }

    // --- on_device + ScalarNative ---

    #[test]
    fn test_powf_tensor_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_scalar("exponent", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: f32) -> Tensor<B, 2> {
            let output = base.powf_scalar(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_tensor_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::F32)
            .input_scalar("exponent", DType::I32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2>, exponent: i32) -> Tensor<B, 2> {
            let output = base.powi_scalar(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_int_tensor_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_tensor("base", 2, DType::I32)
            .input_scalar("exponent", DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: Tensor<B, 2, Int>, exponent: i32) -> Tensor<B, 2, Int> {
            let output = base.powi_scalar(exponent);
            output
        }
        ");
    }

    // --- ScalarNative + ScalarNative ---

    #[test]
    fn test_powf_scalar_native_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::F32)
            .input_scalar("exponent", DType::F32)
            .output_scalar("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: f32, exponent: f32) -> f32 {
            let output = base.powf(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_scalar_native_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::F32)
            .input_scalar("exponent", DType::I32)
            .output_scalar("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: f32, exponent: i32) -> f32 {
            let output = base.powi(exponent as i32);
            output
        }
        ");
    }

    #[test]
    fn test_pow_int_scalar_native_scalar_native() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::I32)
            .input_scalar("exponent", DType::I32)
            .output_scalar("output", DType::I32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: i32, exponent: i32) -> i32 {
            let output = base.pow(exponent as u32);
            output
        }
        ");
    }

    // --- ScalarNative + on_device ---

    #[test]
    fn test_powf_scalar_native_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::F32)
            .input_tensor("exponent", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: f32, exponent: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = Tensor::<
                B,
                1,
            >::from_data_dtype(
                    burn::tensor::TensorData::from([base as f64]),
                    &*self.device,
                    burn::tensor::DType::F32,
                )
                .unsqueeze_dims(&[0isize])
                .powf(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powi_scalar_native_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::F32)
            .input_tensor("exponent", 2, DType::I32)
            .output_tensor("output", 2, DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: f32, exponent: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = Tensor::<
                B,
                1,
            >::from_data_dtype(
                    burn::tensor::TensorData::from([base as f64]),
                    &*self.device,
                    burn::tensor::DType::F32,
                )
                .unsqueeze_dims(&[0isize])
                .powi(exponent);
            output
        }
        ");
    }

    #[test]
    fn test_powf_scalar_native_scalar_tensor() {
        let node = PowNodeBuilder::new("pow1")
            .input_scalar("base", DType::F32)
            .input_scalar_tensor("exponent", DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r"
        pub fn forward(&self, base: f32, exponent: Tensor<B, 1>) -> Tensor<B, 1> {
            let output = Tensor::<
                B,
                1,
            >::from_data_dtype(
                    burn::tensor::TensorData::from([base as f64]),
                    &*self.device,
                    burn::tensor::DType::F32,
                )
                .powf(exponent);
            output
        }
        ");
    }
}
