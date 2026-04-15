extern crate alloc;

use alloc::rc::Rc;
use burn::tensor::Shape;
use burn_store::{TensorSnapshot, TensorSnapshotError};
use proc_macro2::{Ident, Span, TokenStream};

use onnx_ir::Argument;

use crate::burn::BurnImports;

/// A field in the generated model struct
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Ident,
    pub ty: TokenStream,
    pub init: TokenStream,
}

impl Field {
    pub fn new<S: AsRef<str>>(name: S, ty: TokenStream, init: TokenStream) -> Self {
        if name.as_ref().is_empty() {
            panic!("Field with type {ty:?} was passed with empty name");
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

    /// (Optional) Collect tensor snapshots for burnpack serialization.
    ///
    /// Returns tensor snapshots with paths like "{field_name}.weight", "{field_name}.bias".
    /// The snapshots must be lazy - data should only be loaded when `to_data()` is called.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name that will be used as the prefix for tensor paths
    ///
    /// # Notes
    ///
    /// For nodes without learnable parameters, the default implementation returns an empty vec.
    fn collect_snapshots(&self, _field_name: &str) -> Vec<TensorSnapshot> {
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
/// This is commonly used in the forward() method to generate variable names
/// for inputs and outputs.
///
/// # Arguments
///
/// * `arg` - The argument to convert
///
/// # Returns
///
/// A proc_macro2::Ident with the argument's name
pub fn arg_to_ident(arg: &Argument) -> proc_macro2::Ident {
    proc_macro2::Ident::new(&arg.name, proc_macro2::Span::call_site())
}

// ============================================================================
// Tensor snapshot helpers using Flex backend
// ============================================================================

/// The backend used for tensor transformations during ONNX import.
///
/// Flex is a non-generic backend with a *fixed* compile-time `FloatElem = f32`
/// and `IntElem = i32`, but its runtime `TensorData` layer is dtype-dynamic:
/// a `Tensor<Flex, N>` can hold f16/f32/f64/i8–i64/u8–u64/bool data as long as
/// the dtype is pinned at construction time via `Tensor::from_data(data,
/// (&device, dtype))` rather than the bare `&device` overload.
///
/// **Precision caveat:** callers of this type who want to preserve the ONNX
/// weight dtype (especially f64) MUST pass the explicit `(device, dtype)` tuple
/// to `from_data` / `zeros` / `ones`. Using the bare `&device` overload makes
/// `Tensor::from_data` call `resolve_dtype(None) = f32`, which then invokes
/// `data.convert_dtype(F32)` and silently truncates f64 weights before the
/// tensor is ever constructed. The lstm/gru/rnn weight-snapshot helpers in
/// this module all follow the pinned-dtype pattern for exactly this reason.
pub type SerializationBackend = burn::backend::Flex;

/// Create a lazy tensor snapshot from an ONNX argument.
///
/// This creates a TensorSnapshot that lazily loads tensor data only when needed.
/// The closure captures the argument and calls `value()` only when `to_data()` is invoked.
///
/// # Arguments
///
/// * `input` - The ONNX argument containing tensor data
/// * `path` - The tensor path (e.g., "linear1.weight")
/// * `container_type` - The container type (e.g., "Linear")
///
/// # Returns
///
/// A TensorSnapshot with lazy data loading
pub fn create_lazy_snapshot(
    input: &Argument,
    path: &str,
    container_type: &str,
) -> Option<TensorSnapshot> {
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use onnx_ir::ir::ArgType;

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

    // Create a lazy closure that only loads data when called
    let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
        let mut data = input_clone.value().ok_or_else(|| {
            TensorSnapshotError::DataError(format!(
                "Failed to extract tensor data for '{}'",
                input_clone.name
            ))
        })?;
        // Scalar data has shape [], but Param<Tensor<B, 1>> expects shape [1]
        if is_scalar && data.shape.is_empty() {
            data.shape = Shape::from([1]);
        }
        Ok(data)
    });

    // Parse path into path_stack
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    Some(TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_stack,
        container_stack,
        ParamId::new(),
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
}
