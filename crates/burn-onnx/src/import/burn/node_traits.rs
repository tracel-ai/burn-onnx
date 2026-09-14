extern crate alloc;

use burn::tensor::Shape;
use burn_pack::{Error as PackError, Tensor as PackTensor};
use burn_store::bridge;
use proc_macro2::{Ident, Span, TokenStream};

use onnx_ir::Argument;

use crate::burn::BurnImports;

/// A field in the generated model struct
#[derive(Debug, Clone)]
pub struct Field {
    /// Field name in the generated struct.
    pub name: Ident,
    /// Field type as tokens, e.g. `Linear<B>`.
    pub ty: TokenStream,
    /// Initialization expression assigned to the field in `new()`.
    pub init: TokenStream,
}

impl Field {
    /// Create a field from a name, type tokens, and initialization tokens.
    ///
    /// Panics if `name` is empty, or if it was built by stringifying an
    /// `arg_to_ident()` result: the struct field would be renamed when the
    /// file is written while the burnpack weight path kept the tag.
    pub fn new<S: AsRef<str>>(name: S, ty: TokenStream, init: TokenStream) -> Self {
        if name.as_ref().is_empty() {
            panic!("Field with type {ty:?} was passed with empty name");
        }
        if super::shadow_check::is_tagged_name(name.as_ref()) {
            panic!(
                "Field name {:?} was built from a graph value ident; use the argument's name, \
                 not arg_to_ident()",
                name.as_ref()
            );
        }
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            ty,
            init,
        }
    }
}

/// Tensor kind (Int, Float, Bool)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Int,
    Float,
    Bool,
}

impl From<onnx_ir::ir::DType> for TensorKind {
    fn from(dtype: onnx_ir::ir::DType) -> Self {
        if dtype.is_float() {
            TensorKind::Float
        } else if dtype.is_int() || dtype.is_uint() {
            TensorKind::Int
        } else if dtype.is_bool() {
            TensorKind::Bool
        } else {
            panic!("Unsupported tensor type: {dtype:?}")
        }
    }
}

impl quote::ToTokens for TensorKind {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        // TODO use this throughout the codebase
        let kind = match self {
            TensorKind::Int => quote::quote! { Int },
            TensorKind::Float => quote::quote! { Float },
            TensorKind::Bool => quote::quote! { Bool },
        };
        tokens.extend(kind);
    }
}

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen: std::fmt::Debug {
    /// Returns all input arguments for this node.
    ///
    /// # Notes
    ///
    /// This should return ALL inputs, including static initializers.
    /// Filtering (e.g., for dynamic/constant inputs only) is done at the call site.
    fn inputs(&self) -> &[Argument];

    /// Returns all output arguments for this node.
    ///
    /// # Notes
    ///
    /// This should return ALL outputs.
    fn outputs(&self) -> &[Argument];

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [ScopeAtPosition](super::scope::ScopeAtPosition) encapsulates both the scope and node position.
    /// Use `scope.arg()` to automatically handle Tensor/Scalar/Shape arguments with proper clone tracking.
    fn forward(&self, scope: &mut super::scope::ScopeAtPosition<'_>) -> TokenStream;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

    /// (Optional) Declare the type and initialization of the field
    ///
    /// # Notes
    ///
    /// This should be implemented when the node has some parameters.
    /// Just one field per type is possible, if the node has multiple types for its parameters, a
    /// tuple can be used.
    ///
    /// The returned Field struct contains both the type and initialization code.
    fn field(&self) -> Option<Field> {
        None
    }

    /// (Optional) Collect deferred tensors for burnpack serialization.
    ///
    /// Returns deferred tensors with paths like "{field_name}.weight", "{field_name}.bias".
    /// The tensors must stay deferred - bytes should only be produced when the writer
    /// requests them.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name that will be used as the prefix for tensor paths
    ///
    /// # Notes
    ///
    /// For nodes without learnable parameters, the default implementation returns an empty vec.
    fn collect_tensors(&self, _field_name: &str) -> Vec<PackTensor> {
        vec![]
    }
}

// ============================================================================
// Node utilities
// ============================================================================

/// Helper function to extract tensor data from a node input.
///
/// This is commonly used by nodes that need to access constant tensor values
/// (e.g., weights, biases, normalization parameters).
///
/// # Arguments
///
/// * `inputs` - The node's input arguments
/// * `input_index` - Index of the input to extract data from
///
/// # Returns
///
/// `Some(TensorData)` if the input has a constant value, `None` otherwise
pub fn extract_node_data(
    inputs: &[onnx_ir::Argument],
    input_index: usize,
) -> Option<burn::tensor::TensorData> {
    let input = inputs.get(input_index)?;
    input.value()
}

/// Helper function to convert an Argument's name to a proc_macro2::Ident.
///
/// Use it for output bindings and host-side values in `forward()`.
///
/// The returned ident is tagged for `shadow_check` (`__arg_<name>`) and the
/// tag is stripped when the model file is written. Only splice it into
/// generated tokens: stringifying it or deriving another identifier from it
/// (`format_ident!("{}_len", ident)`) leaks the tag. Use `arg.name` for that.
///
/// # Arguments
///
/// * `arg` - The argument to convert
///
/// # Returns
///
/// The tagged ident for the argument
pub fn arg_to_ident(arg: &Argument) -> proc_macro2::Ident {
    super::shadow_check::value_ident(&arg.name)
}

// ============================================================================
// Deferred tensor helpers
// ============================================================================
//
// Tensors created during ONNX import (e.g., for slicing weight blobs in the
// rnn/lstm/gru tensor helpers) MUST pin the runtime dtype via
// `Tensor::from_data(data, (&device, dtype))` rather than the bare `&device`
// overload. The bare form lets `Tensor::from_data` resolve the dtype from the
// device's default `FloatDType`, which can silently truncate f64 weights to
// f32 before they enter the tensor pipeline.

/// Create a deferred burnpack tensor from an ONNX argument.
///
/// The returned tensor carries only metadata until its bytes are drawn: the closure
/// captures the argument and calls `value()` only when the writer asks for the data.
/// That keeps a save's peak memory bounded by the largest single tensor.
///
/// # Arguments
///
/// * `input` - The ONNX argument containing tensor data
/// * `path` - The tensor path (e.g., "linear1.weight")
///
/// # Returns
///
/// A deferred [`PackTensor`], or `None` when the input carries no static data
pub fn create_deferred_tensor(input: &Argument, path: &str) -> Option<PackTensor> {
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use onnx_ir::ir::ArgType;

    // Skip Dynamic and Optional inputs: there is no static data to capture.
    // Constant inputs are intentionally let through so an unlifted constant
    // fails loudly here rather than being silently dropped (which would
    // produce a model with zero-initialized weights at load time).
    if input.is_dynamic() || input.is_optional() {
        return None;
    }

    // Get tensor metadata without loading data
    let (dtype, shape, is_scalar) = match &input.ty {
        ArgType::Tensor(tensor_type) => {
            let dtype = tensor_type.dtype;
            let shape: Shape = tensor_type.static_shape_known().unwrap_or_default().into();
            (dtype, shape, false)
        }
        ArgType::ScalarTensor(d) | ArgType::ScalarNative(d) => (*d, Shape::from([1]), true),
        _ => return None,
    };

    // Clone the input for the closure (lightweight, doesn't copy tensor data)
    let input_clone = input.clone();

    // Only loads data when the writer draws the bytes.
    Some(bridge::deferred(
        path.to_string(),
        dtype,
        shape,
        Some(ParamId::new().val()),
        move || -> Result<TensorData, PackError> {
            let mut data = input_clone.value().ok_or_else(|| {
                PackError::ValidationError(format!(
                    "Failed to extract tensor data for '{}'",
                    input_clone.name
                ))
            })?;
            // Scalar data has shape [], but Param<Tensor<1>> expects shape [1]
            if is_scalar && data.shape.is_empty() {
                data.shape = Shape::from([1]);
            }
            Ok(data)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::ir::{BoolStore, DType};

    #[test]
    fn tensor_kind_from_dtype_float_types() {
        assert_eq!(TensorKind::from(DType::F16), TensorKind::Float);
        assert_eq!(TensorKind::from(DType::BF16), TensorKind::Float);
        assert_eq!(TensorKind::from(DType::F32), TensorKind::Float);
        assert_eq!(TensorKind::from(DType::F64), TensorKind::Float);
    }

    #[test]
    fn tensor_kind_from_dtype_signed_int_types() {
        assert_eq!(TensorKind::from(DType::I8), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::I16), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::I32), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::I64), TensorKind::Int);
    }

    #[test]
    fn tensor_kind_from_dtype_unsigned_int_types() {
        assert_eq!(TensorKind::from(DType::U8), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::U16), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::U32), TensorKind::Int);
        assert_eq!(TensorKind::from(DType::U64), TensorKind::Int);
    }

    #[test]
    fn tensor_kind_from_dtype_bool() {
        assert_eq!(
            TensorKind::from(DType::Bool(BoolStore::Native)),
            TensorKind::Bool
        );
    }

    #[test]
    fn create_deferred_tensor_skips_dynamic_input() {
        use onnx_ir::ir::{ArgType, Argument, TensorType, ValueSource};

        let arg = Argument::new(
            "slope",
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 1,
                static_shape: Some(vec![Some(3)]),
            }),
        );
        assert_eq!(arg.value_source, ValueSource::Dynamic);
        assert!(create_deferred_tensor(&arg, "prelu1.alpha").is_none());
    }

    #[test]
    fn create_deferred_tensor_skips_optional_input() {
        use onnx_ir::ir::{ArgType, Argument, TensorType, ValueSource};

        let arg = Argument::new(
            "",
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 1,
                static_shape: None,
            }),
        );
        assert_eq!(arg.value_source, ValueSource::Optional);

        assert!(create_deferred_tensor(&arg, "deform_conv1.bias").is_none());
    }

    /// An unlifted constant fails when the writer draws its bytes, and that failure has to
    /// name both ends: the burnpack path to locate it in the model, and the ONNX argument to
    /// locate it in the graph. The path comes from burn-pack, whose writer annotates every
    /// provider error with the tensor it was drawing; the argument name comes from the
    /// closure here. Neither alone is enough to act on.
    #[test]
    fn unlifted_constant_write_failure_names_path_and_argument() {
        use onnx_ir::ir::{ArgType, Argument, TensorType, ValueSource};

        let mut arg = Argument::new(
            "onnx_initializer_7",
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 1,
                static_shape: Some(vec![Some(3)]),
            }),
        );
        // Claims to be a constant, but nothing ever lifted a value into the store.
        arg.value_source = ValueSource::Constant;

        let tensor = create_deferred_tensor(&arg, "conv1.weight").expect("a constant is captured");

        let err = burn_pack::Writer::new(vec![tensor])
            .into_bytes()
            .expect_err("an unlifted constant must fail the write");

        let msg = err.to_string();
        assert!(msg.contains("conv1.weight"), "missing burnpack path: {msg}");
        assert!(
            msg.contains("onnx_initializer_7"),
            "missing ONNX argument name: {msg}"
        );
    }
}
