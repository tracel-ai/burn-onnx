use super::prelude::*;

impl NodeCodegen for onnx_ir::expand::ExpandNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        let input = scope.arg(input_arg);
        let output = arg_to_ident(output_arg);
        let output_rank = output_arg.ty.rank();

        // ONNX right-aligns a `shape` shorter than the input, so the requested dims are
        // padded on the left with 1s up to the output rank; max-semantics below then keeps
        // the input's leading dims.
        let shape = match &self.config {
            onnx_ir::expand::ExpandConfig::Static(s) => {
                let mut padded = vec![1i64; output_rank.saturating_sub(s.len())];
                padded.extend(s);
                padded.to_tokens()
            }
            onnx_ir::expand::ExpandConfig::Runtime(r) => {
                let shape_arg = &self.inputs[r.input_index];
                let (shape_len, value) = match &shape_arg.ty {
                    ArgType::Tensor(tensor) => {
                        // The length is only known from a static shape. Without one it is
                        // assumed to equal the output rank; a shorter runtime shape then
                        // panics in the `try_into` below instead of being padded.
                        let shape_len = tensor
                            .static_shape
                            .as_ref()
                            .and_then(|dims| dims.first().copied().flatten())
                            .unwrap_or(output_rank);
                        let name = arg_to_ident(shape_arg);
                        let value = quote! {
                            TryInto::<[i64; #shape_len]>::try_into(
                                #name.to_data().convert::<i64>().as_slice().unwrap()
                            ).unwrap()
                        };
                        (shape_len, value)
                    }
                    ArgType::Shape(shape_len) => {
                        let name = arg_to_ident(shape_arg);
                        (*shape_len, quote! { #name })
                    }
                    other => {
                        unreachable!("Expand shape input type validated in onnx-ir, got {other:?}")
                    }
                };
                if shape_len < output_rank {
                    let pad = output_rank - shape_len;
                    quote! {
                        {
                            let requested: [i64; #shape_len] = #value;
                            let mut padded = [1i64; #output_rank];
                            padded[#pad..].copy_from_slice(&requested);
                            padded
                        }
                    }
                } else {
                    value
                }
            }
        };

        // For Shape inputs, convert [i64; N] array to a 1D Int tensor and expand.
        if let ArgType::Shape(shape_rank) = &input_arg.ty {
            let shape_rank = shape_rank.to_tokens();
            let shape_dim_offset = dim_offset(output_rank, 1);
            return quote! {
                let #output = {
                    let onnx_shape: [i64; #output_rank] = #shape;
                    let input_tensor = Tensor::<1, Int>::from_data(
                        burn::tensor::TensorData::from(#input.as_slice()),
                        (&self.device, burn::tensor::DType::I64)
                    );
                    let input_dims = [#shape_rank];
                    let mut shape = onnx_shape;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..1usize {
                        let dim_offset = #shape_dim_offset;
                        if shape[dim_offset] == 1 && input_dims[i] > 1 {
                            shape[dim_offset] = input_dims[i] as i64;
                        }
                    }
                    input_tensor.expand(shape)
                };
            };
        }

        // ScalarTensor is already a Tensor<1> on device, just expand directly.
        if input_arg.ty.is_scalar_tensor() {
            let static_shape = match &self.config {
                onnx_ir::expand::ExpandConfig::Static(s) => Some(s.clone()),
                _ => None,
            };
            // Empty shape means identity (keep same shape)
            if static_shape.as_ref().is_some_and(|s| s.is_empty()) {
                return quote! {
                    let #output = #input;
                };
            }
            let rank_tok = output_rank.to_tokens();
            return quote! {
                let #output = {
                    let shape: [i64; #rank_tok] = #shape;
                    #input.expand(shape)
                };
            };
        }

        // For ScalarNative inputs, materialize as rank-1 tensor and expand.
        if input_arg.ty.is_scalar_native() {
            let dtype = input_arg.ty.elem_type();
            let dtype_tokens = dtype.to_tokens();
            let kind = match dtype {
                DType::Bool(_) => quote! { , Bool },
                _ if dtype.is_float() => quote! {},
                _ => quote! { , Int },
            };
            return quote! {
                let #output = {
                    let input = Tensor::<1 #kind>::from_data(
                        burn::tensor::TensorData::from([#input]),
                        (&self.device, #dtype_tokens)
                    );
                    input.expand(#shape)
                };
            };
        }

        // ONNX Expand uses max-semantics: output_dim = max(input_dim, shape_dim)
        let input_rank = input_arg.ty.rank();
        let dim_offset = dim_offset(output_rank, input_rank);
        quote! {
            let #output = {
                let onnx_shape: [i64; #output_rank] = #shape;
                let input_dims = #input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..#input_rank {
                    let dim_offset = #dim_offset;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                #input.expand(shape)
            };
        }
    }
}

/// Index expression `output_rank - input_rank + i` for the right-aligned broadcast loop,
/// with the rank difference folded at codegen time.
///
/// Emitting the subtraction verbatim produces `3usize - 3usize + i` when the ranks match,
/// which trips clippy's deny-by-default `eq_op` lint in the generated code.
fn dim_offset(output_rank: usize, input_rank: usize) -> TokenStream {
    let Some(offset) = output_rank.checked_sub(input_rank) else {
        unreachable!(
            "Expand output rank is max(input rank, shape length) in onnx-ir, \
             got output rank {output_rank} < input rank {input_rank}"
        )
    };
    match offset {
        0 => quote! { i },
        offset => quote! { #offset + i },
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::expand::{ExpandConfig, ExpandNode, ExpandNodeBuilder};
    use onnx_ir::ir::RuntimeInputRef;

    fn create_expand_node_static(name: &str, input_rank: usize, shape: Vec<i64>) -> ExpandNode {
        let output_rank = shape.len();
        let config = ExpandConfig::Static(shape);

        ExpandNodeBuilder::new(name)
            .input_tensor("input", input_rank, DType::F32)
            .output_tensor("output", output_rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_expand_static() {
        let node = create_expand_node_static("expand1", 2, vec![2, 3, 4]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [2, 3, 4];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..2usize {
                    let dim_offset = 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_broadcast() {
        let node = create_expand_node_static("expand1", 2, vec![1, 5, 10]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [1, 5, 10];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..2usize {
                    let dim_offset = 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_same_rank() {
        // Equal ranks fold the offset away instead of emitting `3usize - 3usize + i`,
        // which clippy's deny-by-default `eq_op` rejects.
        let node = create_expand_node_static("expand1", 3, vec![2, 3, 4]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [2, 3, 4];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..3usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_static_shorter_shape() {
        // A shape shorter than the input is right-aligned: padded with leading 1s so the
        // input's leading dims are kept.
        let node = ExpandNodeBuilder::new("expand1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(ExpandConfig::Static(vec![4]))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = [1, 1, 4];
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..3usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_runtime_shorter_shape() {
        let node = ExpandNodeBuilder::new("expand1")
            .input_tensor("input", 3, DType::F32)
            .input_shape("shape", 1)
            .output_tensor("output", 3, DType::F32)
            .config(ExpandConfig::Runtime(RuntimeInputRef::new(
                "shape".to_string(),
                1,
            )))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, shape: [i64; 1]) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = {
                    let requested: [i64; 1usize] = shape;
                    let mut padded = [1i64; 3usize];
                    padded[2usize..].copy_from_slice(&requested);
                    padded
                };
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..3usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_runtime_tensor_shorter_shape() {
        // The shape tensor's static length (1) is shorter than the output rank, so the
        // values read at runtime are left-padded with 1s.
        let node = ExpandNodeBuilder::new("expand1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor_shape("shape", vec![1], DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(ExpandConfig::Runtime(RuntimeInputRef::new(
                "shape".to_string(),
                1,
            )))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, shape: Tensor<1, Int>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = {
                    let requested: [i64; 1usize] = TryInto::<
                        [i64; 1usize],
                    >::try_into(shape.to_data().convert::<i64>().as_slice().unwrap())
                        .unwrap();
                    let mut padded = [1i64; 3usize];
                    padded[2usize..].copy_from_slice(&requested);
                    padded
                };
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..3usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_runtime_tensor_unknown_length() {
        // Without a static length the shape tensor is read as the full output rank.
        let node = ExpandNodeBuilder::new("expand1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("shape", 1, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(ExpandConfig::Runtime(RuntimeInputRef::new(
                "shape".to_string(),
                1,
            )))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, shape: Tensor<1, Int>) -> Tensor<3> {
            let output = {
                let onnx_shape: [i64; 3usize] = TryInto::<
                    [i64; 3usize],
                >::try_into(shape.to_data().convert::<i64>().as_slice().unwrap())
                    .unwrap();
                let input_dims = input.dims();
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..3usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_int64() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: i64) -> Tensor<2, Int> {
            let output = {
                let input = Tensor::<
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from([input]),
                    (&self.device, burn::tensor::DType::I64),
                );
                input.expand([2, 3])
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_f32() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> Tensor<2> {
            let output = {
                let input = Tensor::<
                    1,
                >::from_data(
                    burn::tensor::TensorData::from([input]),
                    (&self.device, burn::tensor::DType::F32),
                );
                input.expand([2, 3])
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_bool() {
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_scalar")
            .input_scalar("input", DType::Bool(BoolStore::Native))
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: bool) -> Tensor<2, Bool> {
            let output = {
                let input = Tensor::<
                    1,
                    Bool,
                >::from_data(
                    burn::tensor::TensorData::from([input]),
                    (&self.device, burn::tensor::DType::Bool(burn::tensor::BoolStore::Native)),
                );
                input.expand([2, 3])
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_scalar_tensor_i64() {
        // ScalarTensor is already on device, just expand directly
        let config = ExpandConfig::Static(vec![2, 3]);
        let node = ExpandNodeBuilder::new("expand_st")
            .input_scalar_tensor("input", DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<1, Int>) -> Tensor<2, Int> {
            let output = {
                let shape: [i64; 2] = [2, 3];
                input.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_shape_as_data() {
        // Shape input (1D int64 array) expanded to a target shape.
        // This pattern occurs in piper-tts/VITS (issue #266).
        let config = ExpandConfig::Static(vec![2, 2]);
        let node = ExpandNodeBuilder::new("expand1")
            .input_shape("shape_out", 2)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_out: [i64; 2]) -> Tensor<2, Int> {
            let output = {
                let onnx_shape: [i64; 2usize] = [2, 2];
                let input_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(shape_out.as_slice()),
                    (&self.device, burn::tensor::DType::I64),
                );
                let input_dims = [2];
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..1usize {
                    let dim_offset = 1usize + i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input_tensor.expand(shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_expand_shape_as_data_rank1_output() {
        // Rank-1 output from a Shape input: the offset folds to `i` rather than
        // `1usize - 1usize + i`.
        let config = ExpandConfig::Static(vec![3]);
        let node = ExpandNodeBuilder::new("expand1")
            .input_shape("shape_out", 3)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_out: [i64; 3]) -> Tensor<1, Int> {
            let output = {
                let onnx_shape: [i64; 1usize] = [3];
                let input_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(shape_out.as_slice()),
                    (&self.device, burn::tensor::DType::I64),
                );
                let input_dims = [3];
                let mut shape = onnx_shape;
                #[allow(clippy::needless_range_loop)]
                for i in 0..1usize {
                    let dim_offset = i;
                    if shape[dim_offset] == 1 && input_dims[i] > 1 {
                        shape[dim_offset] = input_dims[i] as i64;
                    }
                }
                input_tensor.expand(shape)
            };
            output
        }
        ");
    }
}
