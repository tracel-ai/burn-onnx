//! ONNX argument types
//!
//! This module contains types for representing node inputs and outputs,
//! including their types, data sources, and metadata.

use core::fmt;
use std::fmt::Formatter;

use burn_tensor::DType;

use super::tensor_data_ext::TensorData;
use crate::tensor_store::ValueStore;

pub type Rank = usize;
pub type Shape = Vec<Option<usize>>;

/// Unique identifier for tensor data in the central store
pub type DataId = usize;

/// Describes where an argument's value comes from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSource {
    /// Static constant value embedded in the argument (name="" with embedded data)
    Static(DataId),
    /// Points to a constant node output (name="constant1_out1")
    Constant,
    /// Points to a runtime node output (name="conv1_out1")
    Dynamic,
    /// Optional/not provided (name="")
    Optional,
}

/// A node input or output.
#[derive(Clone)]
pub struct Argument {
    /// The name of the node input.
    pub name: String,

    /// The type of the argument.
    pub ty: ArgType,

    /// Describes where this argument's value comes from
    /// For Static values, contains the tensor data ID directly
    pub value_source: ValueSource,

    /// Reference to value storage for constant lookup
    /// This is an Rc-wrapped immutable store - no RefCell needed since we only read
    pub(crate) value_store: Option<ValueStore>,
}

impl fmt::Debug for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Argument")
            .field("name", &self.name)
            .field("ty", &self.ty)
            .field("value_source", &self.value_source)
            .finish()
    }
}

impl Argument {
    /// Copy everything except the name from the other argument
    pub fn copy_value(&mut self, other_arg: &Argument) {
        self.ty = other_arg.ty.clone();
        self.value_source = other_arg.value_source;
    }
}

/// The type of an argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgType {
    Scalar(DType),
    Shape(Rank),
    Tensor(TensorType),
}

/// Represents the type of a tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    /// The data type of the tensor values (e.g. F32, F64, I64, etc.)
    pub dtype: DType,

    /// The number of dimensions in the tensor
    pub rank: Rank,

    /// Static shape if known (populated during shape inference).
    /// Each dimension is independently `Some(val)` or `None` (symbolic).
    pub static_shape: Option<Vec<Option<usize>>>,
}

impl Default for TensorType {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            rank: 0,
            static_shape: None,
        }
    }
}

impl TensorType {
    pub fn new(dtype: DType, rank: Rank, static_shape: Option<Vec<Option<usize>>>) -> Self {
        Self {
            dtype,
            rank,
            static_shape,
        }
    }

    /// Create a TensorType with a fully-known static shape (all dims concrete).
    pub fn new_known(dtype: DType, shape: Vec<usize>) -> Self {
        let rank = shape.len();
        Self {
            dtype,
            rank,
            static_shape: Some(shape.into_iter().map(Some).collect()),
        }
    }

    /// Returns the static shape only if ALL dimensions are known (no symbolic dims).
    pub fn static_shape_known(&self) -> Option<Vec<usize>> {
        self.static_shape
            .as_ref()
            .and_then(|dims| dims.iter().copied().collect::<Option<Vec<usize>>>())
    }

    /// Merge static_shape from another TensorType, keeping the most informative value
    /// for each dimension. Returns true if any dimension was updated.
    ///
    /// Merge rules:
    /// - If self has no shape info but other does, use other's shape
    /// - If both have shape info with matching rank, merge dimension by dimension
    ///   (prefer known values over unknown)
    /// - If ranks differ, no merge is possible (other types like dtype/rank take precedence)
    pub fn merge_static_shape(&mut self, other: &TensorType) -> bool {
        // Ranks must match for any merge to be valid
        if self.rank != other.rank {
            return false;
        }

        match (&mut self.static_shape, &other.static_shape) {
            // self has no shape info, other does -> take other's shape
            (None, Some(other_shape)) if other_shape.len() == self.rank => {
                self.static_shape = Some(other_shape.clone());
                true
            }
            // Both have shape info with matching length -> merge dimension by dimension
            (Some(self_shape), Some(other_shape)) if self_shape.len() == other_shape.len() => {
                let mut changed = false;
                for (self_dim, other_dim) in self_shape.iter_mut().zip(other_shape.iter()) {
                    // If self doesn't know this dimension but other does, take other's value
                    if self_dim.is_none() && other_dim.is_some() {
                        *self_dim = *other_dim;
                        changed = true;
                    }
                }
                changed
            }
            // Other cases: no merge possible or nothing to merge
            _ => false,
        }
    }
}

impl Default for ArgType {
    fn default() -> Self {
        Self::Tensor(TensorType::default())
    }
}

impl ArgType {
    /// Merge static_shape information from another ArgType.
    ///
    /// Used during type inference to preserve shape info from value_info when
    /// syncing inferred types. The inferred type (self) provides the base
    /// dtype/rank, but we merge in any more-specific dimension info from other.
    ///
    /// Returns true if any shape dimension was updated.
    pub fn merge_static_shape(&mut self, other: &ArgType) -> bool {
        match (self, other) {
            (ArgType::Tensor(self_tensor), ArgType::Tensor(other_tensor)) => {
                self_tensor.merge_static_shape(other_tensor)
            }
            // Non-tensor types have no static_shape to merge
            _ => false,
        }
    }

    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// Check if this is a tensor type
    pub fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }

    /// Check if this is a shape type
    pub fn is_shape(&self) -> bool {
        matches!(self, Self::Shape(_))
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        match self {
            ArgType::Scalar(_) => 0,
            ArgType::Shape(_) => 1,
            ArgType::Tensor(t) => t.rank,
        }
    }

    //TODO Element kind

    /// Get the data type
    pub fn elem_type(&self) -> DType {
        match self {
            ArgType::Scalar(s) => *s,
            ArgType::Shape(_) => panic!("ArgType::Shape has no DType"),
            ArgType::Tensor(t) => t.dtype,
        }
    }

    /// Get the static shape if available (per-dim, may contain `None` for symbolic dims)
    pub fn static_shape(&self) -> Option<&Vec<Option<usize>>> {
        match self {
            ArgType::Tensor(t) => t.static_shape.as_ref(),
            _ => None,
        }
    }

    /// Get the static shape only if ALL dimensions are known (no symbolic dims)
    pub fn static_shape_known(&self) -> Option<Vec<usize>> {
        match self {
            ArgType::Tensor(t) => t.static_shape_known(),
            _ => None,
        }
    }
}

impl Argument {
    /// Create a new argument with a specific type
    pub fn new(name: impl Into<String>, ty: ArgType) -> Self {
        let name = name.into();
        // Default to Dynamic (points to a node output by name)
        let value_source = if name.is_empty() {
            ValueSource::Optional
        } else {
            ValueSource::Dynamic
        };

        Self {
            name,
            ty,
            value_source,
            value_store: None,
        }
    }

    /// Create a new argument with default type (F32 tensor rank 0)
    pub fn from_name(name: impl Into<String>) -> Self {
        Self::new(name, ArgType::default())
    }

    /// Get the constant value from the central tensor store
    pub fn value(&self) -> Option<TensorData> {
        if self.value_store.is_none() {
            if matches!(
                self.value_source,
                ValueSource::Constant | ValueSource::Static(_)
            ) {
                log::warn!(
                    "value() called on '{}' (value_source={:?}) but value_store is None",
                    self.name,
                    self.value_source
                );
            }
            return None;
        }
        let store = self.value_store.as_ref()?;

        match &self.value_source {
            // Static: data is embedded directly
            ValueSource::Static(data_id) => {
                let result = store.get_tensor_data(*data_id);
                if result.is_none() {
                    log::warn!(
                        "value() for Static({}) on '{}' returned None - store has {} tensors",
                        data_id,
                        self.name,
                        store.tensor_count()
                    );
                }
                result
            }
            // Constant: look up the constant node by output name
            ValueSource::Constant => {
                let data_id = store.get_constant_data_id(&self.name);
                if data_id.is_none() {
                    log::warn!(
                        "value() lookup failed for '{}': constant not found in store (constant_map has {} entries)",
                        self.name,
                        store.constant_map_len()
                    );
                }
                let data_id = data_id?;
                store.get_tensor_data(data_id)
            }
            // Dynamic/Optional: no constant data
            ValueSource::Dynamic | ValueSource::Optional => None,
        }
    }

    /// Set the value store
    pub(crate) fn set_value_store(&mut self, store: ValueStore) {
        self.value_store = Some(store);
    }

    /// Check if this is a static constant (embedded value)
    pub fn is_static(&self) -> bool {
        matches!(self.value_source, ValueSource::Static(_))
    }

    /// Check if this argument points to a constant node output
    pub fn is_constant(&self) -> bool {
        self.value_source == ValueSource::Constant
    }

    /// Check if this argument points to a runtime node output
    pub fn is_dynamic(&self) -> bool {
        self.value_source == ValueSource::Dynamic
    }

    /// Check if this argument is optional/not provided
    pub fn is_optional(&self) -> bool {
        self.value_source == ValueSource::Optional
    }

    /// Convert a Constant argument to Static by embedding the constant's data
    ///
    /// This looks up the constant node by name, retrieves its data_id,
    /// and embeds it in this argument, clearing the name.
    ///
    /// Returns an error if this is not a Constant argument.
    pub fn to_static(&mut self) -> Result<(), crate::processor::ProcessError> {
        use crate::processor::ProcessError;

        if !self.is_constant() {
            return Err(ProcessError::Custom(format!(
                "Cannot convert {:?} argument to Static (only Constant can be converted)",
                self.value_source
            )));
        }

        // Look up the constant node by name
        let store = self.value_store.as_ref().ok_or_else(|| {
            ProcessError::Custom("No value store available to look up constant".to_string())
        })?;

        // Get the data_id from the constant map using the output name
        let data_id = store.get_constant_data_id(&self.name).ok_or_else(|| {
            ProcessError::Custom(format!(
                "Constant node not found or has no data_id for output name: {}",
                self.name
            ))
        })?;

        // Embed the data_id in ValueSource::Static, clear the name
        // The name is cleared because Static values are accessed via data_id, not by name
        self.value_source = ValueSource::Static(data_id);
        self.name.clear();

        Ok(())
    }

    /// Create an argument with a constant i64 scalar value embedded.
    pub fn from_const_i64(name: impl Into<String>, value: i64) -> Self {
        Self::from_const_i64s(name, &[value], ArgType::Scalar(DType::I64))
    }

    /// Create an argument with a constant 1D i64 tensor embedded.
    pub fn from_const_i64_shape(name: impl Into<String>, values: &[i64]) -> Self {
        Self::from_const_i64s(name, values, ArgType::Shape(values.len()))
    }

    fn from_const_i64s(name: impl Into<String>, values: &[i64], ty: ArgType) -> Self {
        use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};
        use std::collections::HashMap;
        use std::rc::Rc;

        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let shape = if values.len() == 1 && matches!(ty, ArgType::Scalar(_)) {
            vec![]
        } else {
            vec![values.len()]
        };
        let data_ref = TensorDataRef::new(bytes::Bytes::from(bytes), shape, DType::I64);

        let mut tensor_store = TensorStore::new();
        let data_id = tensor_store.store(data_ref);

        let value_store = ValueStore::new(Rc::new(tensor_store), Rc::new(HashMap::new()));

        Self {
            name: name.into(),
            ty,
            value_source: ValueSource::Static(data_id),
            value_store: Some(value_store),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_static_shape_none_with_some() {
        // Inferred type has no shape info, value_info has partial shape
        let mut inferred = TensorType::new(DType::F32, 2, None);
        let value_info = TensorType::new(DType::F32, 2, Some(vec![None, Some(1)]));

        let changed = inferred.merge_static_shape(&value_info);

        assert!(changed);
        assert_eq!(inferred.static_shape, Some(vec![None, Some(1)]));
    }

    #[test]
    fn test_merge_static_shape_some_with_none() {
        // Inferred type has shape info, value_info has none
        let mut inferred = TensorType::new(DType::F32, 2, Some(vec![Some(5), None]));
        let value_info = TensorType::new(DType::F32, 2, None);

        let changed = inferred.merge_static_shape(&value_info);

        assert!(!changed);
        assert_eq!(inferred.static_shape, Some(vec![Some(5), None]));
    }

    #[test]
    fn test_merge_static_shape_dimension_by_dimension() {
        // Both have partial info, different dimensions known
        let mut inferred = TensorType::new(DType::F32, 2, Some(vec![Some(5), None]));
        let value_info = TensorType::new(DType::F32, 2, Some(vec![None, Some(1)]));

        let changed = inferred.merge_static_shape(&value_info);

        assert!(changed);
        assert_eq!(inferred.static_shape, Some(vec![Some(5), Some(1)]));
    }

    #[test]
    fn test_merge_static_shape_no_change_when_same() {
        // Both have the same info
        let mut inferred = TensorType::new(DType::F32, 2, Some(vec![Some(5), Some(1)]));
        let value_info = TensorType::new(DType::F32, 2, Some(vec![Some(5), Some(1)]));

        let changed = inferred.merge_static_shape(&value_info);

        assert!(!changed);
        assert_eq!(inferred.static_shape, Some(vec![Some(5), Some(1)]));
    }

    #[test]
    fn test_merge_static_shape_inferred_takes_precedence_when_both_known() {
        // Both know a dimension but with different values - inferred wins
        let mut inferred = TensorType::new(DType::F32, 2, Some(vec![Some(5), Some(2)]));
        let value_info = TensorType::new(DType::F32, 2, Some(vec![Some(10), Some(1)]));

        let changed = inferred.merge_static_shape(&value_info);

        // Inferred already has values, so no change
        assert!(!changed);
        assert_eq!(inferred.static_shape, Some(vec![Some(5), Some(2)]));
    }

    #[test]
    fn test_merge_static_shape_rank_mismatch_no_merge() {
        // Ranks don't match - can't merge
        let mut inferred = TensorType::new(DType::F32, 3, None);
        let value_info = TensorType::new(DType::F32, 2, Some(vec![None, Some(1)]));

        let changed = inferred.merge_static_shape(&value_info);

        assert!(!changed);
        assert_eq!(inferred.static_shape, None);
    }

    #[test]
    fn test_merge_argtype_tensor() {
        let mut inferred =
            ArgType::Tensor(TensorType::new(DType::F32, 2, Some(vec![Some(5), None])));
        let value_info = ArgType::Tensor(TensorType::new(DType::F32, 2, Some(vec![None, Some(1)])));

        let changed = inferred.merge_static_shape(&value_info);

        assert!(changed);
        assert_eq!(
            inferred,
            ArgType::Tensor(TensorType::new(DType::F32, 2, Some(vec![Some(5), Some(1)])))
        );
    }

    #[test]
    fn test_merge_argtype_non_tensor_no_op() {
        let mut inferred = ArgType::Scalar(DType::F32);
        let value_info = ArgType::Scalar(DType::F32);

        let changed = inferred.merge_static_shape(&value_info);

        assert!(!changed);
    }
}
