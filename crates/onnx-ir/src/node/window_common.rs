//! Shared helpers for ONNX window generation operators (HammingWindow, HannWindow).
//!
//! These ops have identical shape: one scalar integer `size` input, `periodic` and
//! `output_datatype` attributes, and a 1-D window tensor output. Only the mathematical
//! formula differs. The shared types and helpers live here so HammingWindow and HannWindow
//! can stay in sync.

use crate::ir::{DType, RawNode, RuntimeInputRef};
use crate::processor::ProcessError;
use crate::proto_conversion::element_type_from_proto;

/// Represents either a static or runtime window size.
#[derive(Debug, Clone)]
pub enum WindowSize {
    /// Size known at compile time.
    Static(usize),
    /// Size determined at runtime from a graph input.
    Runtime(RuntimeInputRef),
}

impl Default for WindowSize {
    fn default() -> Self {
        Self::Static(0)
    }
}

/// Resolve the `output_datatype` attribute (default: FLOAT).
///
/// ONNX spec constrains the output to float types (F16, BF16, F32, F64). Integer
/// types are rejected with `InvalidAttribute` because a window function produces
/// coefficients in `[0, 1]` that would silently truncate to mostly zeros.
pub(crate) fn resolve_output_dtype(node: &RawNode, op_name: &str) -> Result<DType, ProcessError> {
    let dtype = match node.attrs.get("output_datatype") {
        Some(val) => {
            let dt_i32 = val.clone().into_i32();
            element_type_from_proto(dt_i32).map_err(|e| ProcessError::InvalidAttribute {
                name: "output_datatype".to_string(),
                reason: format!("{op_name}: {e}"),
            })?
        }
        None => DType::F32,
    };

    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        return Err(ProcessError::InvalidAttribute {
            name: "output_datatype".to_string(),
            reason: format!("{op_name}: must be a float type, got {dtype:?}"),
        });
    }

    Ok(dtype)
}
