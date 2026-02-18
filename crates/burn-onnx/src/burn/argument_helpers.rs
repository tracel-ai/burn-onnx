//! Helper functions for working with onnx_ir::Argument types
//!
//! This module provides utilities to generate code for different argument types
//! without needing the Type abstraction layer.

use onnx_ir::{
    Argument,
    ir::{ArgType, DType},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::burn::ToTokens;

/// Get the type TokenStream for an argument
pub fn arg_type_tokens(arg: &Argument) -> TokenStream {
    match &arg.ty {
        ArgType::Tensor(tensor) => {
            let rank = tensor.rank.to_tokens();
            match &tensor.dtype {
                dtype if dtype.is_float() => quote! { Tensor<B, #rank> },
                dtype if dtype.is_int() || dtype.is_uint() => {
                    quote! { Tensor<B, #rank, Int> }
                }
                dtype if dtype.is_bool() => quote! { Tensor<B, #rank, Bool> },
                _ => quote! { Tensor<B, #rank> },
            }
        }
        ArgType::ScalarNative(dtype) => scalar_type_tokens(dtype),
        ArgType::ScalarTensor(dtype) => match dtype {
            d if d.is_float() => quote! { Tensor<B, 1> },
            d if d.is_int() || d.is_uint() => quote! { Tensor<B, 1, Int> },
            d if d.is_bool() => quote! { Tensor<B, 1, Bool> },
            _ => quote! { Tensor<B, 1> },
        },
        ArgType::Shape(rank) => {
            let rank_lit = rank.to_tokens();
            quote! { [i64; #rank_lit] }
        }
    }
}

/// Get the type TokenStream for a scalar DType
pub fn scalar_type_tokens(dtype: &DType) -> TokenStream {
    match dtype {
        DType::F16 => quote! { half::f16 },
        DType::BF16 => quote! { half::bf16 },
        DType::F32 => quote! { f32 },
        DType::F64 => quote! { f64 },
        DType::I8 => quote! { i8 },
        DType::I16 => quote! { i16 },
        DType::I32 => quote! { i32 },
        DType::I64 => quote! { i64 },
        DType::U8 => quote! { u8 },
        DType::U16 => quote! { u16 },
        DType::U32 => quote! { u32 },
        DType::U64 => quote! { u64 },
        DType::Bool => quote! { bool },
        _ => panic!("Unsupported scalar dtype: {:?}", dtype),
    }
}

/// Generate `.elem::<T>()` cast for extracting a native scalar from `into_scalar()`.
///
/// Used after `.into_scalar()` to cast from the backend's native element type
/// to the desired Rust type.
pub fn elem_cast_tokens(dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! { .elem::<#ty>() }
}

/// Generate code to extract a native scalar from a on-device tensor.
///
/// Produces: `<input>.into_scalar().elem::<T>()`
pub fn on_device_to_native(input: TokenStream, dtype: &DType) -> TokenStream {
    let cast = elem_cast_tokens(dtype);
    quote! { #input.into_scalar()#cast }
}

/// Generate code to convert a ScalarTensor (Tensor<B,1>) to a Shape([i64; 1]).
///
/// Produces: `{ let __v: T = <input>.into_scalar().elem(); [__v as i64] }`
pub fn scalar_tensor_to_shape(input: TokenStream, dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! {
        {
            let __v: #ty = #input.into_scalar().elem();
            [__v as i64]
        }
    }
}

/// Generate code to convert a native scalar to a Shape([i64; 1]).
///
/// Produces: `[<input> as i64]`
pub fn scalar_native_to_shape(input: TokenStream) -> TokenStream {
    quote! { [#input as i64] }
}

/// Generate code to extract the first element of a Shape as a native scalar.
///
/// Produces: `<input>[0] as T`
pub fn shape_to_native(input: TokenStream, dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! { #input[0] as #ty }
}

/// Get the argument identifier
pub fn arg_ident(arg: &Argument) -> Ident {
    Ident::new(&arg.name, Span::call_site())
}

/// Generate function parameters from a slice of arguments
///
/// Produces: `name1: Type1, name2: Type2, ...`
pub fn codegen_fn_params(args: &[Argument]) -> TokenStream {
    let params: Vec<_> = args
        .iter()
        .map(|arg| {
            let name = arg_ident(arg);
            let ty = arg_type_tokens(arg);
            quote! { #name: #ty }
        })
        .collect();

    quote! { #(#params),* }
}

/// Generate return type from output arguments
///
/// Single output: `Type`
/// Multiple outputs: `(Type1, Type2, ...)`
pub fn codegen_return_type(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        arg_type_tokens(&outputs[0])
    } else {
        let types: Vec<_> = outputs.iter().map(arg_type_tokens).collect();
        quote! { (#(#types),*) }
    }
}

/// Generate return expression from output arguments
///
/// Single output: `name`
/// Multiple outputs: `(name1, name2, ...)`
pub fn codegen_return_expr(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        let name = arg_ident(&outputs[0]);
        quote! { #name }
    } else {
        let names: Vec<_> = outputs.iter().map(arg_ident).collect();
        quote! { (#(#names),*) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_type_tokens_float_types() {
        assert_eq!(
            scalar_type_tokens(&DType::F16).to_string(),
            quote!(half::f16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::BF16).to_string(),
            quote!(half::bf16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::F32).to_string(),
            quote!(f32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::F64).to_string(),
            quote!(f64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_signed_int_types() {
        assert_eq!(
            scalar_type_tokens(&DType::I8).to_string(),
            quote!(i8).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I16).to_string(),
            quote!(i16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I32).to_string(),
            quote!(i32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I64).to_string(),
            quote!(i64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_unsigned_int_types() {
        assert_eq!(
            scalar_type_tokens(&DType::U8).to_string(),
            quote!(u8).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U16).to_string(),
            quote!(u16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U32).to_string(),
            quote!(u32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U64).to_string(),
            quote!(u64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_bool() {
        assert_eq!(
            scalar_type_tokens(&DType::Bool).to_string(),
            quote!(bool).to_string()
        );
    }
}
