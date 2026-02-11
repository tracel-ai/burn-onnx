use proc_macro2::TokenStream;
use quote::quote;

use onnx_ir::ir::DType;
use onnx_ir::node::padding::{AutoPad, PaddingConfig1d, PaddingConfig2d, PaddingConfig3d};

// ============================================================================
// Codegen utilities for converting types to TokenStream
// ============================================================================

fn convert_primitive<T: core::fmt::Debug>(primitive: T) -> TokenStream {
    let value = format!("{primitive:?}");

    value.parse().unwrap()
}

fn convert_to_array<'a, I, T>(list: I) -> TokenStream
where
    I: Iterator<Item = &'a T>,
    T: ToTokens + 'a,
{
    let mut body = quote! {};

    list.for_each(|item| {
        let elem = item.to_tokens();
        body.extend(quote! {#elem,});
    });

    quote! {
        [#body]
    }
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

impl<const N: usize, T: Copy + ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

impl<T: Copy + ToTokens> ToTokens for Vec<T> {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

/// Prettier output for `usize`
impl ToTokens for usize {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `i64`
impl ToTokens for i64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f64`
impl ToTokens for f64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f32`
impl ToTokens for f32 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Convert an f32 value to tokens, handling non-finite values (inf, NaN)
/// that `proc_macro2::Literal` cannot represent as literals.
pub fn f32_to_tokens(val: f32) -> TokenStream {
    if val.is_nan() {
        quote! { f32::NAN }
    } else if val == f32::INFINITY {
        quote! { f32::INFINITY }
    } else if val == f32::NEG_INFINITY {
        quote! { f32::NEG_INFINITY }
    } else {
        // Use proc_macro2 Literal directly to get a suffixed float (e.g. 3.14f32)
        let lit = proc_macro2::Literal::f32_suffixed(val);
        quote! { #lit }
    }
}

/// Convert an f64 value to tokens, handling non-finite values (inf, NaN)
/// that `proc_macro2::Literal` cannot represent as literals.
pub fn f64_to_tokens(val: f64) -> TokenStream {
    if val.is_nan() {
        quote! { f64::NAN }
    } else if val == f64::INFINITY {
        quote! { f64::INFINITY }
    } else if val == f64::NEG_INFINITY {
        quote! { f64::NEG_INFINITY }
    } else {
        // Use proc_macro2 Literal directly to get a suffixed float (e.g. 2.718f64)
        let lit = proc_macro2::Literal::f64_suffixed(val);
        quote! { #lit }
    }
}

/// Padding configuration for 1D operations.
///
/// Converts PaddingConfig1d to Rust code tokens.
/// Format: Explicit(left, right)
impl ToTokens for PaddingConfig1d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig1d::Valid },
            Self::Explicit(left, right) => {
                let left = left.to_tokens();
                let right = right.to_tokens();
                quote! { PaddingConfig1d::Explicit(#left, #right) }
            }
        }
    }
}

/// Converts PaddingConfig2d to Rust code tokens.
/// Format: Explicit(top, left, bottom, right)
impl ToTokens for PaddingConfig2d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig2d::Valid },
            Self::Explicit(top, left, bottom, right) => {
                let top = top.to_tokens();
                let left = left.to_tokens();
                let bottom = bottom.to_tokens();
                let right = right.to_tokens();
                quote! { PaddingConfig2d::Explicit(#top, #left, #bottom, #right) }
            }
        }
    }
}

/// Converts PaddingConfig3d to Rust code tokens.
/// Burn only supports symmetric 3D padding: Explicit(depth, height, width).
/// Asymmetric 3D padding will cause a codegen-time panic.
impl ToTokens for PaddingConfig3d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig3d::Valid },
            Self::Explicit(front, top, left, back, bottom, right) => {
                if self.is_asymmetric() {
                    panic!(
                        "Asymmetric 3D padding is not supported by Burn. \
                         Got front={front}, top={top}, left={left}, back={back}, bottom={bottom}, right={right}"
                    );
                }
                let depth = front.to_tokens();
                let height = top.to_tokens();
                let width = left.to_tokens();
                quote! { PaddingConfig3d::Explicit(#depth, #height, #width) }
            }
        }
    }
}

// ============================================================================
// Auto-pad computation helpers
// ============================================================================

/// Compute padding for one spatial dimension from ONNX auto_pad semantics.
///
/// Returns (pad_begin, pad_end).
fn compute_auto_pad_1dim(
    auto_pad: &AutoPad,
    input_size: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    match auto_pad {
        AutoPad::Valid => (0, 0),
        AutoPad::SameUpper | AutoPad::SameLower => {
            let effective_kernel = (kernel - 1) * dilation + 1;
            let output_size = input_size.div_ceil(stride); // ceil(input/stride)
            let total_pad =
                ((output_size - 1) * stride + effective_kernel).saturating_sub(input_size);
            let pad_small = total_pad / 2;
            let pad_big = total_pad - pad_small;
            match auto_pad {
                AutoPad::SameUpper => (pad_small, pad_big),
                AutoPad::SameLower => (pad_big, pad_small),
                _ => unreachable!(),
            }
        }
        AutoPad::NotSet => panic!("compute_auto_pad_1dim called with NotSet"),
    }
}

/// Resolve auto_pad to a PaddingConfig1d.
///
/// Panics if auto_pad is SameUpper/SameLower and input_spatial is None.
pub fn resolve_auto_pad_1d(
    auto_pad: &AutoPad,
    padding: &PaddingConfig1d,
    input_spatial: Option<&[usize]>,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> PaddingConfig1d {
    match auto_pad {
        AutoPad::NotSet => padding.clone(),
        AutoPad::Valid => PaddingConfig1d::Valid,
        AutoPad::SameUpper | AutoPad::SameLower => {
            let shape = input_spatial
                .expect("auto_pad SAME_UPPER/SAME_LOWER requires static input shape, but input has dynamic dimensions. Use explicit pads instead");
            let (left, right) = compute_auto_pad_1dim(auto_pad, shape[0], kernel, stride, dilation);
            PaddingConfig1d::Explicit(left, right)
        }
    }
}

/// Resolve auto_pad to a PaddingConfig2d.
pub fn resolve_auto_pad_2d(
    auto_pad: &AutoPad,
    padding: &PaddingConfig2d,
    input_spatial: Option<&[usize]>,
    kernel: &[usize; 2],
    stride: &[usize; 2],
    dilation: &[usize; 2],
) -> PaddingConfig2d {
    match auto_pad {
        AutoPad::NotSet => padding.clone(),
        AutoPad::Valid => PaddingConfig2d::Valid,
        AutoPad::SameUpper | AutoPad::SameLower => {
            let shape = input_spatial
                .expect("auto_pad SAME_UPPER/SAME_LOWER requires static input shape, but input has dynamic dimensions. Use explicit pads instead");
            let (top, bottom) =
                compute_auto_pad_1dim(auto_pad, shape[0], kernel[0], stride[0], dilation[0]);
            let (left, right) =
                compute_auto_pad_1dim(auto_pad, shape[1], kernel[1], stride[1], dilation[1]);
            PaddingConfig2d::Explicit(top, left, bottom, right)
        }
    }
}

/// Resolve auto_pad to a PaddingConfig3d.
pub fn resolve_auto_pad_3d(
    auto_pad: &AutoPad,
    padding: &PaddingConfig3d,
    input_spatial: Option<&[usize]>,
    kernel: &[usize; 3],
    stride: &[usize; 3],
    dilation: &[usize; 3],
) -> PaddingConfig3d {
    match auto_pad {
        AutoPad::NotSet => padding.clone(),
        AutoPad::Valid => PaddingConfig3d::Valid,
        AutoPad::SameUpper | AutoPad::SameLower => {
            let shape = input_spatial
                .expect("auto_pad SAME_UPPER/SAME_LOWER requires static input shape, but input has dynamic dimensions. Use explicit pads instead");
            let (front, back) =
                compute_auto_pad_1dim(auto_pad, shape[0], kernel[0], stride[0], dilation[0]);
            let (top, bottom) =
                compute_auto_pad_1dim(auto_pad, shape[1], kernel[1], stride[1], dilation[1]);
            let (left, right) =
                compute_auto_pad_1dim(auto_pad, shape[2], kernel[2], stride[2], dilation[2]);
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_pad_same_upper_symmetric() {
        // input=7, kernel=3, stride=1, dilation=1
        // effective_kernel=3, output=ceil(7/1)=7, total_pad=max(0,6*1+3-7)=2
        // begin=1, end=1
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::SameUpper, 7, 3, 1, 1);
        assert_eq!((begin, end), (1, 1));
    }

    #[test]
    fn test_auto_pad_same_upper_asymmetric() {
        // input=7, kernel=4, stride=1, dilation=1
        // effective_kernel=4, output=7, total_pad=max(0,6+4-7)=3
        // begin=1, end=2
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::SameUpper, 7, 4, 1, 1);
        assert_eq!((begin, end), (1, 2));
    }

    #[test]
    fn test_auto_pad_same_lower_asymmetric() {
        // Same as above but SAME_LOWER flips
        // begin=2, end=1
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::SameLower, 7, 4, 1, 1);
        assert_eq!((begin, end), (2, 1));
    }

    #[test]
    fn test_auto_pad_valid() {
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::Valid, 7, 3, 1, 1);
        assert_eq!((begin, end), (0, 0));
    }

    #[test]
    fn test_auto_pad_with_stride() {
        // input=7, kernel=3, stride=2, dilation=1
        // effective_kernel=3, output=ceil(7/2)=4, total_pad=max(0,3*2+3-7)=2
        // begin=1, end=1
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::SameUpper, 7, 3, 2, 1);
        assert_eq!((begin, end), (1, 1));
    }

    #[test]
    fn test_auto_pad_with_dilation() {
        // input=7, kernel=3, stride=1, dilation=2
        // effective_kernel=(3-1)*2+1=5, output=7, total_pad=max(0,6+5-7)=4
        // begin=2, end=2
        let (begin, end) = compute_auto_pad_1dim(&AutoPad::SameUpper, 7, 3, 1, 2);
        assert_eq!((begin, end), (2, 2));
    }

    #[test]
    fn test_resolve_auto_pad_1d_not_set() {
        let padding = PaddingConfig1d::Explicit(1, 2);
        let result = resolve_auto_pad_1d(&AutoPad::NotSet, &padding, None, 3, 1, 1);
        assert_eq!(result, PaddingConfig1d::Explicit(1, 2));
    }

    #[test]
    fn test_resolve_auto_pad_1d_same_upper() {
        let result = resolve_auto_pad_1d(
            &AutoPad::SameUpper,
            &PaddingConfig1d::Valid,
            Some(&[7]),
            3,
            1,
            1,
        );
        assert_eq!(result, PaddingConfig1d::Explicit(1, 1));
    }

    #[test]
    fn test_resolve_auto_pad_2d_same_upper() {
        let result = resolve_auto_pad_2d(
            &AutoPad::SameUpper,
            &PaddingConfig2d::Valid,
            Some(&[7, 7]),
            &[3, 3],
            &[1, 1],
            &[1, 1],
        );
        assert_eq!(result, PaddingConfig2d::Explicit(1, 1, 1, 1));
    }

    #[test]
    fn test_f32_to_tokens_finite() {
        let tokens = f32_to_tokens(3.14f32);
        assert_eq!(tokens.to_string(), "3.14f32");
    }

    #[test]
    fn test_f32_to_tokens_infinity() {
        let tokens = f32_to_tokens(f32::INFINITY);
        assert_eq!(tokens.to_string(), "f32 :: INFINITY");
    }

    #[test]
    fn test_f32_to_tokens_neg_infinity() {
        let tokens = f32_to_tokens(f32::NEG_INFINITY);
        assert_eq!(tokens.to_string(), "f32 :: NEG_INFINITY");
    }

    #[test]
    fn test_f32_to_tokens_nan() {
        let tokens = f32_to_tokens(f32::NAN);
        assert_eq!(tokens.to_string(), "f32 :: NAN");
    }

    #[test]
    fn test_f64_to_tokens_finite() {
        let tokens = f64_to_tokens(2.718f64);
        assert_eq!(tokens.to_string(), "2.718f64");
    }

    #[test]
    fn test_f64_to_tokens_infinity() {
        let tokens = f64_to_tokens(f64::INFINITY);
        assert_eq!(tokens.to_string(), "f64 :: INFINITY");
    }

    #[test]
    fn test_f64_to_tokens_neg_infinity() {
        let tokens = f64_to_tokens(f64::NEG_INFINITY);
        assert_eq!(tokens.to_string(), "f64 :: NEG_INFINITY");
    }

    #[test]
    fn test_f64_to_tokens_nan() {
        let tokens = f64_to_tokens(f64::NAN);
        assert_eq!(tokens.to_string(), "f64 :: NAN");
    }

    #[test]
    fn test_resolve_auto_pad_3d_valid() {
        let result = resolve_auto_pad_3d(
            &AutoPad::Valid,
            &PaddingConfig3d::Explicit(1, 1, 1, 1, 1, 1),
            None,
            &[3, 3, 3],
            &[1, 1, 1],
            &[1, 1, 1],
        );
        assert_eq!(result, PaddingConfig3d::Valid);
    }
}

/// DType for specifying tensor element types in generated code.
///
/// Note: Flex32 and QFloat are intentionally not supported as they are Burn-specific
/// runtime types that cannot come from ONNX models. Flex32 is a GPU optimization type
/// and QFloat requires quantization schemes not representable in ONNX.
impl ToTokens for DType {
    fn to_tokens(&self) -> TokenStream {
        match self {
            DType::F16 => quote! { burn::tensor::DType::F16 },
            DType::BF16 => quote! { burn::tensor::DType::BF16 },
            DType::F32 => quote! { burn::tensor::DType::F32 },
            DType::F64 => quote! { burn::tensor::DType::F64 },
            DType::I8 => quote! { burn::tensor::DType::I8 },
            DType::I16 => quote! { burn::tensor::DType::I16 },
            DType::I32 => quote! { burn::tensor::DType::I32 },
            DType::I64 => quote! { burn::tensor::DType::I64 },
            DType::U8 => quote! { burn::tensor::DType::U8 },
            DType::U16 => quote! { burn::tensor::DType::U16 },
            DType::U32 => quote! { burn::tensor::DType::U32 },
            DType::U64 => quote! { burn::tensor::DType::U64 },
            DType::Bool => quote! { burn::tensor::DType::Bool },
            // Flex32 and QFloat are Burn-specific runtime types not present in ONNX models
            _ => panic!(
                "Unsupported dtype for ONNX code generation: {:?}. \
                 Flex32 and QFloat are Burn-specific runtime types.",
                self
            ),
        }
    }
}
