use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::ir::TensorDataExt;

impl NodeCodegen for onnx_ir::node::constant::ConstantNode {
    fn inputs(&self) -> &[Argument] {
        // Constant has no runtime inputs - data comes from the input's value store
        &[]
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let output = self.outputs.first().unwrap();

        let (rank, dtype, is_scalar_tensor) = match &output.ty {
            ArgType::Tensor(t) => (t.rank, &t.dtype, false),
            ArgType::ScalarTensor(d) => (1, d, true),
            ArgType::ScalarNative(_) | ArgType::Shape(_) => return None,
        };

        let name = Ident::new(&self.name, Span::call_site());
        let rank_tok = rank.to_tokens();

        let input = self.inputs.first().unwrap();
        let tensor_data = input.value().expect("Constant node must have tensor data");
        let shape = if tensor_data.shape.is_empty() {
            vec![1usize].to_tokens()
        } else {
            tensor_data.shape.to_tokens()
        };

        // For ScalarTensor, embed the actual value in the initializer so Model::new()
        // works without burnpack loading. For regular tensors, use zeros (burnpack loads data).
        // Note: ScalarTensor uses from_data (adopts backend default precision) since these are
        // Param fields. Boundary conversions in graph.rs use from_data_dtype to preserve the
        // exact dtype the internal graph expects from user-provided values.
        let (ty, init) = if is_scalar_tensor {
            // Generate initializer with the actual scalar value
            if dtype.is_float() {
                let v = tensor_data.scalar_f64().unwrap();
                let val = super::super::codegen::f64_to_tokens(v);
                (
                    quote! { burn::module::Param<Tensor<B, 1>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, 1>::from_data([#val], device),
                            device.clone(),
                            false,
                            [1].into(),
                        );
                    },
                )
            } else if dtype.is_int() || dtype.is_uint() {
                let val = tensor_data.to_i64_vec().unwrap()[0];
                (
                    quote! { burn::module::Param<Tensor<B, 1, Int>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([#val], device),
                            device.clone(),
                            false,
                            [1].into(),
                        );
                    },
                )
            } else if dtype.is_bool() {
                let val = tensor_data.as_slice::<bool>().unwrap()[0];
                (
                    quote! { burn::module::Param<Tensor<B, 1, Bool>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, 1, Bool>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, 1, Bool>::from_data([#val], device),
                            device.clone(),
                            false,
                            [1].into(),
                        );
                    },
                )
            } else {
                panic!(
                    "Unsupported ScalarTensor dtype {:?} for constant '{}'",
                    dtype, self.name
                )
            }
        } else {
            // Regular tensor: initialize with zeros, burnpack loads the actual data
            match dtype {
                d if d.is_int() || d.is_uint() => (
                    quote! { burn::module::Param<Tensor<B, #rank_tok, Int>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, #rank_tok, Int>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank_tok, Int>::zeros(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
                        );
                    },
                ),
                d if d.is_float() => (
                    quote! { burn::module::Param<Tensor<B, #rank_tok>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, #rank_tok>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank_tok>::zeros(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
                        );
                    },
                ),
                d if d.is_bool() => (
                    quote! { burn::module::Param<Tensor<B, #rank_tok, Bool>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, #rank_tok, Bool>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank_tok, Bool>::empty(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
                        );
                    },
                ),
                _ => (
                    quote! { burn::module::Param<Tensor<B, #rank_tok>> },
                    quote! {
                        let #name: burn::module::Param<Tensor<B, #rank_tok>> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank_tok>::zeros(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
                        );
                    },
                ),
            }
        };
        Some(Field::new(self.name.clone(), ty, init))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let output = self.outputs.first().unwrap();

        // Collect snapshots for tensor and ScalarTensor constants.
        // ScalarTensor values are also embedded in the field initializer for Model::new(),
        // but burnpack needs them too for Model::from_file() / from_embedded().
        match &output.ty {
            ArgType::Tensor(_) | ArgType::ScalarTensor(_) => {
                if let Some(input) = self.inputs.first() {
                    // Use the field name as the path since constants are stored as single params
                    if let Some(snapshot) = create_lazy_snapshot(input, field_name, "Constant") {
                        vec![snapshot]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            }
            ArgType::ScalarNative(_) | ArgType::Shape(_) => vec![],
        }
    }

    fn forward(&self, _scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let output_ty = &self.outputs.first().unwrap().ty;

        match output_ty {
            ArgType::Tensor(_) | ArgType::ScalarTensor(_) => {
                // For tensor and scalar-tensor constants, reference the stored param
                let name = Ident::new(&self.name, Span::call_site());
                quote! {
                    let #output = self.#name.val();
                }
            }
            ArgType::ScalarNative(elem_type) => {
                // For native scalar constants, embed the value directly as a Rust literal
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");

                let value = match elem_type {
                    onnx_ir::ir::DType::F32 => {
                        let val = tensor_data.as_slice::<f32>().unwrap()[0];
                        super::super::codegen::f32_to_tokens(val)
                    }
                    onnx_ir::ir::DType::F64 => {
                        let val = tensor_data.as_slice::<f64>().unwrap()[0];
                        super::super::codegen::f64_to_tokens(val)
                    }
                    onnx_ir::ir::DType::F16 => {
                        let val = tensor_data.scalar_f64().unwrap();
                        let val_tokens = super::super::codegen::f64_to_tokens(val);
                        quote! { half::f16::from_f64(#val_tokens) }
                    }
                    onnx_ir::ir::DType::BF16 => {
                        let val = tensor_data.scalar_f64().unwrap();
                        let val_tokens = super::super::codegen::f64_to_tokens(val);
                        quote! { half::bf16::from_f64(#val_tokens) }
                    }
                    onnx_ir::ir::DType::I32 => {
                        let val = tensor_data.as_slice::<i32>().unwrap()[0];
                        quote! { #val }
                    }
                    onnx_ir::ir::DType::I64 => {
                        let val = tensor_data.as_slice::<i64>().unwrap()[0];
                        quote! { #val }
                    }
                    d if d.is_int() || d.is_uint() => {
                        // I8, I16, U8, U16, U32, U64
                        let val = tensor_data.to_i64_vec().unwrap()[0];
                        let ty = super::super::argument_helpers::scalar_type_tokens(elem_type);
                        quote! { #val as #ty }
                    }
                    onnx_ir::ir::DType::Bool => {
                        let val = tensor_data.as_slice::<bool>().unwrap()[0];
                        quote! { #val }
                    }
                    _ => panic!(
                        "Unsupported ScalarNative dtype {:?} for constant '{}'",
                        elem_type, self.name
                    ),
                };

                quote! {
                    let #output = #value;
                }
            }
            ArgType::Shape(rank) => {
                // For shape constants, get the shape values from input
                let input = self.inputs.first().unwrap();
                let tensor_data = input.value().expect("Constant node must have tensor data");
                let shape_vec = tensor_data.to_i64_vec().unwrap();

                let values: Vec<_> = shape_vec
                    .iter()
                    .map(|&v| {
                        let v_lit = proc_macro2::Literal::i64_suffixed(v);
                        quote! { #v_lit }
                    })
                    .collect();

                let rank_lit = proc_macro2::Literal::usize_unsuffixed(*rank);

                quote! {
                    let #output: [i64; #rank_lit] = [#(#values),*];
                }
            }
        }
    }
}
