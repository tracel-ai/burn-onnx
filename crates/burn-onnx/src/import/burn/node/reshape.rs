use super::prelude::*;

impl NodeCodegen for onnx_ir::reshape::ReshapeNode {
    fn inputs(&self) -> &[Argument] {
        // Reshape has input tensor and shape argument
        // Filter to include dynamic and constant inputs
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        // Determine if we have static or runtime shape
        match &self.config.shape {
            onnx_ir::reshape::ReshapeInput::Static(shape_values) => {
                let shape_values = reshape_args(&self.config, shape_values.to_tokens());
                // Static shape - simple reshape
                use onnx_ir::ir::ArgType;
                match &input_arg.ty {
                    ArgType::Tensor(_) => {
                        let input = scope.arg(input_arg);

                        // Check if output is a scalar
                        match &output_arg.ty {
                            ArgType::ScalarTensor(_) => {
                                // Keep on device as Tensor<1>
                                quote! {
                                    let #output = #input.reshape([1]);
                                }
                            }
                            ArgType::ScalarNative(elem_type) => {
                                use onnx_ir::ir::DType;
                                let into_scalar = match elem_type {
                                    DType::F32 => quote! { .into_scalar::<f32>() },
                                    DType::F64 => quote! { .into_scalar::<f64>() },
                                    DType::I32 => quote! { .into_scalar::<i32>() },
                                    DType::I64 => quote! { .into_scalar::<i64>() },
                                    DType::Bool(_) => quote! { .into_scalar::<bool>() },
                                    _ => panic!("Unsupported scalar type: {:?}", elem_type),
                                };
                                quote! {
                                    let #output = #input #into_scalar;
                                }
                            }
                            ArgType::Tensor(_) => {
                                quote! {
                                    let #output = #input.reshape(#shape_values);
                                }
                            }
                            ArgType::Shape(_) => {
                                panic!("Tensor to Shape reshape not supported")
                            }
                        }
                    }
                    ArgType::Shape(input_rank) => {
                        // Shape input path
                        let input_name = arg_to_ident(input_arg);

                        match &output_arg.ty {
                            ArgType::ScalarTensor(elem_type) => {
                                if *input_rank != 1 {
                                    panic!(
                                        "Shape to scalar requires Shape(1), got Shape({})",
                                        input_rank
                                    );
                                }
                                // Shape [i64; 1] -> Tensor<1, Int> on device.
                                // Pin the runtime dtype via (device, dtype) so the bare
                                // `&device` overload can't silently narrow the i64 source
                                // to the backend's default IntElem (CLAUDE.md).
                                let dtype_tokens = elem_type.to_tokens();
                                quote! {
                                    let #output = Tensor::<1, Int>::from_data(
                                        burn::tensor::TensorData::from([#input_name[0]]),
                                        (&self.device, #dtype_tokens),
                                    );
                                }
                            }
                            ArgType::ScalarNative(elem_type) => {
                                if *input_rank != 1 {
                                    panic!(
                                        "Shape to scalar requires Shape(1), got Shape({})",
                                        input_rank
                                    );
                                }
                                let cast_expr = shape_to_native(quote! { #input_name }, elem_type);
                                quote! {
                                    let #output = #cast_expr;
                                }
                            }
                            ArgType::Shape(output_rank) => {
                                if input_rank == output_rank {
                                    quote! {
                                        let #output = #input_name;
                                    }
                                } else {
                                    quote! {
                                        let #output: [i64; #output_rank] = {
                                            let mut result = [0i64; #output_rank];
                                            let copy_len = #input_rank.min(#output_rank);
                                            result[..copy_len].copy_from_slice(&#input_name[..copy_len]);
                                            result
                                        };
                                    }
                                }
                            }
                            ArgType::Tensor(output_tensor) => {
                                // Convert Shape to Tensor first, then reshape
                                let dtype_tokens = output_tensor.dtype.to_tokens();
                                quote! {
                                    let #output = {
                                        let shape_array = #input_name as [i64; #input_rank];
                                        Tensor::<1, Int>::from_data(
                                            TensorData::from(shape_array),
                                            (&self.device, #dtype_tokens)
                                        )
                                    }.reshape(#shape_values);
                                }
                            }
                        }
                    }
                    ArgType::ScalarNative(input_dtype) => {
                        // Native scalar input - convert to tensor or pass through
                        let input_name = arg_to_ident(input_arg);

                        match &output_arg.ty {
                            ArgType::Tensor(tensor_type) => {
                                // Create a 1-element tensor from scalar with explicit dtype,
                                // then reshape to target shape (which must also be 1 element total)
                                let output_rank = tensor_type.rank;
                                let dtype_tokens = tensor_type.dtype.to_tokens();

                                if tensor_type.dtype.is_float() {
                                    quote! {
                                        let #output = Tensor::<#output_rank>::from_data(
                                            burn::tensor::TensorData::from([#input_name as f64]),
                                            (&self.device, #dtype_tokens)
                                        ).reshape(#shape_values);
                                    }
                                } else if tensor_type.dtype.is_int() || tensor_type.dtype.is_uint()
                                {
                                    quote! {
                                        let #output = Tensor::<#output_rank, Int>::from_data(
                                            burn::tensor::TensorData::from([#input_name as i64]),
                                            (&self.device, #dtype_tokens)
                                        ).reshape(#shape_values);
                                    }
                                } else {
                                    // Bool output
                                    let bool_expr = if input_dtype.is_bool() {
                                        quote! { #input_name }
                                    } else {
                                        quote! { #input_name != 0 }
                                    };
                                    quote! {
                                        let #output = Tensor::<#output_rank, Bool>::from_data(
                                            burn::tensor::TensorData::from([#bool_expr]),
                                            (&self.device, #dtype_tokens)
                                        ).reshape(#shape_values);
                                    }
                                }
                            }
                            ArgType::ScalarNative(_) => {
                                quote! {
                                    let #output = #input_name;
                                }
                            }
                            ArgType::ScalarTensor(_) => {
                                panic!("Reshape: ScalarNative to ScalarTensor not supported")
                            }
                            _ => {
                                panic!("Reshape: scalar input to {:?} not supported", output_arg.ty)
                            }
                        }
                    }
                    ArgType::ScalarTensor(_) => {
                        // ScalarTensor is already a Tensor<1> on device
                        let input = scope.arg(input_arg);

                        match &output_arg.ty {
                            ArgType::Tensor(_) => {
                                quote! {
                                    let #output = #input.reshape(#shape_values);
                                }
                            }
                            ArgType::ScalarTensor(_) => {
                                quote! {
                                    let #output = #input;
                                }
                            }
                            ArgType::ScalarNative(elem_type) => {
                                let elem_cast = on_device_to_native(input, elem_type);
                                quote! {
                                    let #output = #elem_cast;
                                }
                            }
                            _ => {
                                panic!("Reshape: ScalarTensor to {:?} not supported", output_arg.ty)
                            }
                        }
                    }
                }
            }
            onnx_ir::reshape::ReshapeInput::Runtime(shape_ref) => {
                // Runtime shape - need to extract shape from second input
                let shape_arg = &self.inputs[shape_ref.input_index];
                use onnx_ir::ir::ArgType;

                let input = scope.arg(input_arg);

                match &shape_arg.ty {
                    ArgType::Shape(_) => {
                        let shape_name = arg_to_ident(shape_arg);
                        let shape = reshape_args(&self.config, quote! { #shape_name });
                        quote! {
                            let #output = #input.reshape(#shape);
                        }
                    }
                    ArgType::Tensor(_) => {
                        let shape_name = scope.arg(shape_arg);
                        let output_rank = match &output_arg.ty {
                            ArgType::Tensor(t) => t.rank,
                            // An empty runtime shape reshapes the single element to a scalar,
                            // so the shape values are not needed
                            ArgType::ScalarNative(elem_type) => {
                                let value = on_device_to_native(input, elem_type);
                                return quote! {
                                    let #output = {
                                        let _ = &#shape_name;
                                        #value
                                    };
                                };
                            }
                            _ => panic!("Runtime reshape with tensor shape expects tensor output"),
                        };
                        let array_init = (0..output_rank)
                            .map(|i| {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                quote! { shape_array[#idx] }
                            })
                            .collect::<Vec<_>>();
                        let shape = reshape_args(&self.config, quote! { [#(#array_init),*] });

                        quote! {
                            let #output = {
                                let shape_data = #shape_name.to_data();
                                let shape_array = shape_data.as_slice::<i64>().unwrap();
                                #input.reshape(#shape)
                            };
                        }
                    }
                    ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                        panic!("Reshape: shape argument cannot be scalar")
                    }
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Check if we need TensorData for shape-to-tensor conversion
        if let onnx_ir::ir::ArgType::Shape(_) = &self.inputs.first().unwrap().ty
            && let onnx_ir::ir::ArgType::Tensor(_) = &self.outputs.first().unwrap().ty
        {
            imports.register("burn::tensor::TensorData");
        }
    }
}

fn reshape_args(config: &onnx_ir::reshape::ReshapeConfig, shape: TokenStream) -> TokenStream {
    let may_contain_zero = match &config.shape {
        onnx_ir::reshape::ReshapeInput::Static(dims) => dims.contains(&0),
        onnx_ir::reshape::ReshapeInput::Runtime(_) => true,
    };
    if config.allowzero.unwrap_or(0) == 0 && may_contain_zero {
        quote! { burn::tensor::ZeroDimPlaceholders(#shape) }
    } else {
        shape
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::reshape::{ReshapeConfig, ReshapeInput, ReshapeNodeBuilder};

    // Static Tensor -> Tensor reshapes
    #[test]
    fn test_reshape_static_inferred_dimension() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![2, -1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 3, DType::F32)
            .output_tensor("reshaped", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>) -> Tensor<2> {
            let reshaped = data.reshape([2, -1]);
            reshaped
        }
        ");
    }

    #[test]
    fn test_reshape_static_with_zero_placeholder() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![0, -1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor", 3, DType::F32)
            .output_tensor("result", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<3>) -> Tensor<2> {
            let result = tensor.reshape(burn::tensor::ZeroDimPlaceholders([0, -1]));
            result
        }
        ");
    }

    #[test]
    fn test_reshape_static_literal_zero() {
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(ReshapeConfig {
                shape: ReshapeInput::Static(vec![3, 0]),
                allowzero: Some(1),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<2>) -> Tensor<2> {
            let output = data.reshape([3, 0]);
            output
        }
        ");
    }

    #[test]
    fn test_reshape_runtime_shape_literal_zero() {
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 2, DType::F32)
            .input_shape("target", 2)
            .output_tensor("output", 2, DType::F32)
            .config(ReshapeConfig {
                shape: ReshapeInput::Runtime(RuntimeInputRef::new("target".into(), 1)),
                allowzero: Some(1),
            })
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<2>, target: [i64; 2]) -> Tensor<2> {
            let output = data.reshape(target);
            output
        }
        ");
    }

    // Static Tensor -> Scalar (all scalar types)
    #[test]
    fn test_reshape_tensor_to_scalar_f32() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor", 1, DType::F32)
            .output_scalar("scalar", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<1>) -> f32 {
            let scalar = tensor.into_scalar::<f32>();
            scalar
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_f64() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 1, DType::F64)
            .output_scalar("value", DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1>) -> f64 {
            let value = input.into_scalar::<f64>();
            value
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_i32() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 1, DType::I32)
            .output_scalar("int_val", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<1, Int>) -> i32 {
            let int_val = data.into_scalar::<i32>();
            int_val
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_i64() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 1, DType::I64)
            .output_scalar("long_val", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1, Int>) -> i64 {
            let long_val = input.into_scalar::<i64>();
            long_val
        }
        ");
    }

    #[test]
    fn test_reshape_tensor_to_scalar_bool() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("mask", 1, DType::Bool(BoolStore::Native))
            .output_scalar("flag", DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, mask: Tensor<1, Bool>) -> bool {
            let flag = mask.into_scalar::<bool>();
            flag
        }
        ");
    }

    // Static Shape -> Scalar
    #[test]
    fn test_reshape_shape_to_scalar_i64() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("shape_in", 1)
            .output_scalar("dim", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_in: [i64; 1]) -> i64 {
            let dim = shape_in[0] as i64;
            dim
        }
        ");
    }

    #[test]
    fn test_reshape_shape_to_scalar_i32() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("shape_data", 1)
            .output_scalar("size", DType::I32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_data: [i64; 1]) -> i32 {
            let size = shape_data[0] as i32;
            size
        }
        ");
    }

    // Static Shape -> Shape (same rank)
    #[test]
    fn test_reshape_shape_to_shape_same_rank() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("input_shape", 3)
            .output_shape("output_shape", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input_shape: [i64; 3]) -> [i64; 3] {
            let output_shape = input_shape;
            output_shape
        }
        ");
    }

    // Static Shape -> Shape (different rank)
    #[test]
    fn test_reshape_shape_to_shape_expand() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("small_shape", 2)
            .output_shape("large_shape", 4)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, small_shape: [i64; 2]) -> [i64; 4] {
            let large_shape: [i64; 4usize] = {
                let mut result = [0i64; 4usize];
                let copy_len = 2usize.min(4usize);
                result[..copy_len].copy_from_slice(&small_shape[..copy_len]);
                result
            };
            large_shape
        }
        ");
    }

    #[test]
    fn test_reshape_shape_to_shape_shrink() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("big_shape", 4)
            .output_shape("tiny_shape", 2)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, big_shape: [i64; 4]) -> [i64; 2] {
            let tiny_shape: [i64; 2usize] = {
                let mut result = [0i64; 2usize];
                let copy_len = 4usize.min(2usize);
                result[..copy_len].copy_from_slice(&big_shape[..copy_len]);
                result
            };
            tiny_shape
        }
        ");
    }

    // Static Shape -> Tensor
    #[test]
    fn test_reshape_shape_to_tensor() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![0, 1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("dims", 3)
            .output_tensor("tensor_dims", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, dims: [i64; 3]) -> Tensor<2, Int> {
            let tensor_dims = {
                let shape_array = dims as [i64; 3usize];
                Tensor::<
                    1,
                    Int,
                >::from_data(
                    TensorData::from(shape_array),
                    (&self.device, burn::tensor::DType::I64),
                )
            }
                .reshape(burn::tensor::ZeroDimPlaceholders([0, 1]));
            tensor_dims
        }
        ");
    }

    // Runtime shape with Shape argument
    #[test]
    fn test_reshape_runtime_shape_zero_placeholders() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Runtime(RuntimeInputRef::new("target_shape".into(), 1)),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("data", 3, DType::F32)
            .input_shape("target_shape", 2)
            .output_tensor("reshaped", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>, target_shape: [i64; 2]) -> Tensor<2> {
            let reshaped = data.reshape(burn::tensor::ZeroDimPlaceholders(target_shape));
            reshaped
        }
        ");
    }

    #[test]
    fn test_reshape_runtime_tensor_literal_zero() {
        let config = ReshapeConfig {
            allowzero: Some(1),
            shape: ReshapeInput::Runtime(RuntimeInputRef::new("new_shape".into(), 1)),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("new_shape", 1, DType::I64)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<3>, new_shape: Tensor<1, Int>) -> Tensor<2> {
            let y = {
                let shape_data = new_shape.to_data();
                let shape_array = shape_data.as_slice::<i64>().unwrap();
                x.reshape([shape_array[0], shape_array[1]])
            };
            y
        }
        ");
    }

    // Runtime empty shape with Tensor argument reshapes to a scalar
    #[test]
    fn test_reshape_runtime_with_tensor_to_scalar() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Runtime(RuntimeInputRef::new("new_shape".into(), 1)),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("x", 1, DType::F32)
            .input_tensor("new_shape", 1, DType::I64)
            .output_scalar("y", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>, new_shape: Tensor<1, Int>) -> f32 {
            let y = {
                let _ = &new_shape;
                (x).into_scalar::<f32>()
            };
            y
        }
        ");
    }

    #[test]
    fn test_reshape_runtime_tensor_zero_placeholders() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Runtime(RuntimeInputRef::new("shape_tensor".into(), 1)),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("input", 4, DType::F32)
            .input_tensor("shape_tensor", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, shape_tensor: Tensor<1, Int>) -> Tensor<3> {
            let output = {
                let shape_data = shape_tensor.to_data();
                let shape_array = shape_data.as_slice::<i64>().unwrap();
                input
                    .reshape(
                        burn::tensor::ZeroDimPlaceholders([
                            shape_array[0],
                            shape_array[1],
                            shape_array[2],
                        ]),
                    )
            };
            output
        }
        ");
    }

    // Scalar -> Tensor reshapes
    #[test]
    fn test_reshape_scalar_to_tensor_i64() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![-1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_scalar("value", DType::I64)
            .output_tensor("tensor_out", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, value: i64) -> Tensor<1, Int> {
            let tensor_out = Tensor::<
                1usize,
                Int,
            >::from_data(
                    burn::tensor::TensorData::from([value as i64]),
                    (&self.device, burn::tensor::DType::I64),
                )
                .reshape([-1]);
            tensor_out
        }
        ");
    }

    #[test]
    fn test_reshape_scalar_to_tensor_f32() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![1]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_scalar("val", DType::F32)
            .output_tensor("out", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, val: f32) -> Tensor<1> {
            let out = Tensor::<
                1usize,
            >::from_data(
                    burn::tensor::TensorData::from([val as f64]),
                    (&self.device, burn::tensor::DType::F32),
                )
                .reshape([1]);
            out
        }
        ");
    }

    #[test]
    fn test_reshape_scalar_to_scalar() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_scalar("x", DType::F32)
            .output_scalar("y", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: f32) -> f32 {
            let y = x;
            y
        }
        ");
    }

    // Tensor -> ScalarTensor (keep on device)
    #[test]
    fn test_reshape_tensor_to_scalar_tensor() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_tensor("tensor", 1, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<1>) -> Tensor<1> {
            let output = tensor.reshape([1]);
            output
        }
        ");
    }

    // Shape -> ScalarTensor (put shape value on device)
    #[test]
    fn test_reshape_shape_to_scalar_tensor() {
        let config = ReshapeConfig {
            allowzero: None,
            shape: ReshapeInput::Static(vec![]),
        };
        let node = ReshapeNodeBuilder::new("reshape1")
            .input_shape("shape_in", 1)
            .output_scalar_tensor("output", DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_in: [i64; 1]) -> Tensor<1, Int> {
            let output = Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([shape_in[0]]),
                (&self.device, burn::tensor::DType::I64),
            );
            output
        }
        ");
    }
}
