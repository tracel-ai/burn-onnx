//! Load `TensorProto` `.pb` files shipped with upstream ONNX node tests.
//!
//! ONNX backend tests serialize each input/output tensor as a single
//! `TensorProto` message. This module decodes those files into
//! [`ReferenceTensor`] values covering every dtype that actually appears
//! in the upstream `test_data_set_0/` payloads of ops burn-onnx supports.
//!
//! The dtype coverage is:
//!
//! * Integer: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64
//! * Float:   FLOAT (f32), DOUBLE (f64)
//! * Other:   BOOL
//!
//! Exotic dtypes (FLOAT16, BFLOAT16, FLOAT8*, INT4/UINT4, FLOAT4E2M1,
//! STRING, COMPLEX*) surface as [`LoadError::UnsupportedDataType`]. Every
//! upstream node test that uses them is `skip-codegen` or `fail-compare`
//! in `expectations.toml`, so the harness never tries to load them.
//!
//! ONNX stores values in one of two mutually exclusive places:
//!
//! * The dtype-agnostic `raw_data` field — a little-endian packed byte
//!   buffer of fixed-width scalars. Upstream backend tests use this for
//!   nearly everything, so it is the primary decode path.
//! * A type-specific field (`float_data`, `int32_data`, `int64_data`,
//!   `double_data`, `uint64_data`). We fall back to these for any file
//!   that omits `raw_data`, and pull the bit-exact representation out of
//!   the storage slot the spec assigns to the target dtype (e.g. INT8
//!   lives in `int32_data` as i32 values but must be truncated back to
//!   i8 on read).

use onnx_ir::TensorProto;
use protobuf::Message;
use std::path::{Path, PathBuf};
use thiserror::Error;

// --- ONNX TensorProto.DataType enum values we decode --------------------
//
// Only the variants listed here are handled. The rest are either exotic
// floats/ints that burn-onnx cannot codegen anyway (FLOAT8*, INT4, etc.)
// or sequence/string/complex types that no supported op exercises.

const DATA_TYPE_UNDEFINED: i32 = 0;
const DATA_TYPE_FLOAT: i32 = 1;
const DATA_TYPE_UINT8: i32 = 2;
const DATA_TYPE_INT8: i32 = 3;
const DATA_TYPE_UINT16: i32 = 4;
const DATA_TYPE_INT16: i32 = 5;
const DATA_TYPE_INT32: i32 = 6;
const DATA_TYPE_INT64: i32 = 7;
const DATA_TYPE_BOOL: i32 = 9;
const DATA_TYPE_DOUBLE: i32 = 11;
const DATA_TYPE_UINT32: i32 = 12;
const DATA_TYPE_UINT64: i32 = 13;

/// A decoded upstream reference tensor. The shape is stored once and
/// shared across every dtype variant so callers can do shape-only checks
/// without matching on the payload.
#[derive(Debug, Clone)]
pub struct ReferenceTensor {
    pub shape: Vec<usize>,
    pub values: TensorValues,
}

impl ReferenceTensor {
    /// Total number of scalar elements implied by `shape`.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The ONNX dtype name (e.g. `"FLOAT"`, `"INT64"`) for diagnostics.
    pub fn dtype_name(&self) -> &'static str {
        self.values.dtype_name()
    }
}

/// Typed payload of a [`ReferenceTensor`]. One variant per supported
/// ONNX dtype. Ordering intentionally groups signed/unsigned integers
/// and puts float types last so the enum matches the dtype table in the
/// module docs.
#[derive(Debug, Clone)]
pub enum TensorValues {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TensorValues {
    /// The ONNX dtype name (e.g. `"FLOAT"`) for this variant.
    pub fn dtype_name(&self) -> &'static str {
        match self {
            Self::I8(_) => "INT8",
            Self::I16(_) => "INT16",
            Self::I32(_) => "INT32",
            Self::I64(_) => "INT64",
            Self::U8(_) => "UINT8",
            Self::U16(_) => "UINT16",
            Self::U32(_) => "UINT32",
            Self::U64(_) => "UINT64",
            Self::Bool(_) => "BOOL",
            Self::F32(_) => "FLOAT",
            Self::F64(_) => "DOUBLE",
        }
    }

    /// Element count of the payload, independent of `shape` metadata.
    /// Used by the length-mismatch check to verify the two agree.
    fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }
}

/// A FLOAT-only decoded tensor. Kept for backwards compatibility with
/// the M1 `negative_gate_actually_gates` test; new code should use
/// [`ReferenceTensor`] + [`load_tensor`].
#[derive(Debug, Clone)]
pub struct FloatTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl FloatTensor {
    /// Total number of scalar elements implied by `shape`.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Errors that can occur while loading a `.pb` reference tensor.
///
/// `path` fields are formatted with `{:?}` because `PathBuf` does not
/// implement `Display`; the Debug form is human-readable enough for an
/// error message and avoids pulling in a wrapper type.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("protobuf decode error in {path:?}: {source}")]
    Proto {
        path: PathBuf,
        #[source]
        source: protobuf::Error,
    },
    #[error("{path:?}: TensorProto has UNDEFINED data_type, file is likely malformed or truncated")]
    UndefinedDataType { path: PathBuf },
    #[error(
        "{path:?}: unsupported TensorProto data_type {data_type} \
         (not decoded: FLOAT16, BFLOAT16, FLOAT8*, INT4/UINT4, FLOAT4E2M1, STRING, COMPLEX*)"
    )]
    UnsupportedDataType { path: PathBuf, data_type: i32 },
    #[error("{path:?}: loaded dtype {actual} does not match caller's expected dtype {expected}")]
    DataTypeMismatch {
        path: PathBuf,
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "{path:?}: {dtype} value {value} in type-specific field overflows the target type \
         (expected to fit in the narrower representation)"
    )]
    ValueOverflow {
        path: PathBuf,
        dtype: &'static str,
        value: i64,
    },
    #[error(
        "{path:?}: TensorProto dim[{index}] = {dim} cannot be converted to usize \
         (must be non-negative and fit in the host pointer width)"
    )]
    InvalidDimension {
        path: PathBuf,
        index: usize,
        dim: i64,
    },
    #[error("{path:?}: shape {dims:?} has an element count that overflows usize")]
    ShapeOverflow { path: PathBuf, dims: Vec<i64> },
    #[error(
        "{path:?}: raw_data has {len} bytes, which is not a multiple of {elem_size} \
         and cannot be a buffer of {dtype} values"
    )]
    UnalignedRawData {
        path: PathBuf,
        len: usize,
        elem_size: usize,
        dtype: &'static str,
    },
    #[error(
        "{path:?}: tensor element count mismatch: shape {shape:?} implies {expected} elements, \
         decoded {actual}"
    )]
    LengthMismatch {
        path: PathBuf,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
}

/// Decode a single `TensorProto` `.pb` file into a dtype-tagged
/// [`ReferenceTensor`].
///
/// See the module-level documentation for the supported dtype set.
pub fn load_tensor(path: &Path) -> Result<ReferenceTensor, LoadError> {
    let bytes = std::fs::read(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let proto = TensorProto::parse_from_bytes(&bytes).map_err(|source| LoadError::Proto {
        path: path.to_path_buf(),
        source,
    })?;

    let shape = decode_shape(path, &proto)?;
    let expected = shape.iter().product::<usize>();
    let values = decode_values(path, &proto, expected)?;

    if values.len() != expected {
        return Err(LoadError::LengthMismatch {
            path: path.to_path_buf(),
            shape,
            expected,
            actual: values.len(),
        });
    }

    Ok(ReferenceTensor { shape, values })
}

/// Thin wrapper around [`load_tensor`] that rejects anything other than
/// a FLOAT tensor. Kept so existing M1 tests (notably the negative gate)
/// keep working without being rewritten against the polymorphic API.
pub fn load_float_tensor(path: &Path) -> Result<FloatTensor, LoadError> {
    let tensor = load_tensor(path)?;
    match tensor.values {
        TensorValues::F32(values) => Ok(FloatTensor {
            shape: tensor.shape,
            values,
        }),
        other => Err(LoadError::DataTypeMismatch {
            path: path.to_path_buf(),
            expected: "FLOAT",
            actual: other.dtype_name(),
        }),
    }
}

/// Decode the `dims` field into a validated `Vec<usize>`. Negative or
/// oversized dims are rejected up-front so they cannot silently wrap or
/// overflow downstream.
fn decode_shape(path: &Path, proto: &TensorProto) -> Result<Vec<usize>, LoadError> {
    let mut shape: Vec<usize> = Vec::with_capacity(proto.dims.len());
    let mut expected: usize = 1;
    for (index, &dim) in proto.dims.iter().enumerate() {
        let dim_usize = usize::try_from(dim).map_err(|_| LoadError::InvalidDimension {
            path: path.to_path_buf(),
            index,
            dim,
        })?;
        expected = expected
            .checked_mul(dim_usize)
            .ok_or_else(|| LoadError::ShapeOverflow {
                path: path.to_path_buf(),
                dims: proto.dims.clone(),
            })?;
        shape.push(dim_usize);
    }
    // `expected` is recomputed by the caller, but the checked_mul above
    // is what actually guards against overflow.
    let _ = expected;
    Ok(shape)
}

/// Dispatch to the dtype-specific decoder based on `proto.data_type`.
fn decode_values(
    path: &Path,
    proto: &TensorProto,
    expected: usize,
) -> Result<TensorValues, LoadError> {
    match proto.data_type {
        DATA_TYPE_UNDEFINED => Err(LoadError::UndefinedDataType {
            path: path.to_path_buf(),
        }),
        DATA_TYPE_FLOAT => decode_f32(path, proto).map(TensorValues::F32),
        DATA_TYPE_DOUBLE => decode_f64(path, proto).map(TensorValues::F64),
        DATA_TYPE_INT8 => decode_i8(path, proto, expected).map(TensorValues::I8),
        DATA_TYPE_INT16 => decode_i16(path, proto, expected).map(TensorValues::I16),
        DATA_TYPE_INT32 => decode_i32(path, proto).map(TensorValues::I32),
        DATA_TYPE_INT64 => decode_i64(path, proto).map(TensorValues::I64),
        DATA_TYPE_UINT8 => decode_u8(path, proto, expected).map(TensorValues::U8),
        DATA_TYPE_UINT16 => decode_u16(path, proto, expected).map(TensorValues::U16),
        DATA_TYPE_UINT32 => decode_u32(path, proto).map(TensorValues::U32),
        DATA_TYPE_UINT64 => decode_u64(path, proto).map(TensorValues::U64),
        DATA_TYPE_BOOL => decode_bool(path, proto, expected).map(TensorValues::Bool),
        other => Err(LoadError::UnsupportedDataType {
            path: path.to_path_buf(),
            data_type: other,
        }),
    }
}

// ---- Fixed-width decoders ---------------------------------------------
//
// Each decoder follows the same shape: prefer `raw_data` (little-endian
// packed scalars) and fall back to the type-specific storage field the
// ONNX spec assigns for that dtype. The `raw_data` path validates the
// byte length against the element size so a truncated buffer cannot
// silently parse as a shorter-but-valid sequence.
//
// `expected` is passed only to the sub-byte helpers (i8, u8, bool,
// i16, u16) that have to distinguish between "raw_data packs 1 byte per
// element" and "int32_data stores one i32 per element".

fn decode_f32(path: &Path, proto: &TensorProto) -> Result<Vec<f32>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 4, "FLOAT")?;
        Ok(proto
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        Ok(proto.float_data.clone())
    }
}

fn decode_f64(path: &Path, proto: &TensorProto) -> Result<Vec<f64>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 8, "DOUBLE")?;
        Ok(proto
            .raw_data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    } else {
        Ok(proto.double_data.clone())
    }
}

fn decode_i32(path: &Path, proto: &TensorProto) -> Result<Vec<i32>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 4, "INT32")?;
        Ok(proto
            .raw_data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        Ok(proto.int32_data.clone())
    }
}

fn decode_i64(path: &Path, proto: &TensorProto) -> Result<Vec<i64>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 8, "INT64")?;
        Ok(proto
            .raw_data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    } else {
        Ok(proto.int64_data.clone())
    }
}

fn decode_u32(path: &Path, proto: &TensorProto) -> Result<Vec<u32>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 4, "UINT32")?;
        Ok(proto
            .raw_data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        // UINT32 and UINT64 share the `uint64_data` storage slot; UINT32
        // values are just stored as u64 with the high bits zero.
        proto
            .uint64_data
            .iter()
            .map(|&v| {
                u32::try_from(v).map_err(|_| LoadError::ValueOverflow {
                    path: path.to_path_buf(),
                    dtype: "UINT32",
                    value: v as i64,
                })
            })
            .collect()
    }
}

fn decode_u64(path: &Path, proto: &TensorProto) -> Result<Vec<u64>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 8, "UINT64")?;
        Ok(proto
            .raw_data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    } else {
        Ok(proto.uint64_data.clone())
    }
}

// ---- Sub-32-bit integer and bool decoders -----------------------------
//
// These live in `int32_data` in ONNX's spec-assigned fallback slot, one
// i32 per logical element (even for i8 / u8 / bool). When the file uses
// raw_data, we pack at the native element size (1 byte for i8/u8/bool,
// 2 bytes for i16/u16).

fn decode_i8(path: &Path, proto: &TensorProto, _expected: usize) -> Result<Vec<i8>, LoadError> {
    if !proto.raw_data.is_empty() {
        // i8 has element size 1, so any length is trivially aligned.
        Ok(proto.raw_data.iter().map(|&b| b as i8).collect())
    } else {
        checked_narrow_i32(path, &proto.int32_data, "INT8", |v| i8::try_from(v).ok())
    }
}

fn decode_u8(path: &Path, proto: &TensorProto, _expected: usize) -> Result<Vec<u8>, LoadError> {
    if !proto.raw_data.is_empty() {
        Ok(proto.raw_data.to_vec())
    } else {
        checked_narrow_i32(path, &proto.int32_data, "UINT8", |v| u8::try_from(v).ok())
    }
}

fn decode_i16(path: &Path, proto: &TensorProto, _expected: usize) -> Result<Vec<i16>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 2, "INT16")?;
        Ok(proto
            .raw_data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect())
    } else {
        checked_narrow_i32(path, &proto.int32_data, "INT16", |v| i16::try_from(v).ok())
    }
}

fn decode_u16(path: &Path, proto: &TensorProto, _expected: usize) -> Result<Vec<u16>, LoadError> {
    if !proto.raw_data.is_empty() {
        check_raw_alignment(path, &proto.raw_data, 2, "UINT16")?;
        Ok(proto
            .raw_data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    } else {
        checked_narrow_i32(path, &proto.int32_data, "UINT16", |v| u16::try_from(v).ok())
    }
}

fn decode_bool(
    _path: &Path,
    proto: &TensorProto,
    _expected: usize,
) -> Result<Vec<bool>, LoadError> {
    if !proto.raw_data.is_empty() {
        // ONNX stores BOOL as one byte per element: 0 is false, any
        // non-zero value is true. We treat the raw byte that way rather
        // than asserting `== 1`, both to match ONNX's spec and to avoid
        // rejecting buffers produced by implementations that use a
        // different convention.
        Ok(proto.raw_data.iter().map(|&b| b != 0).collect())
    } else {
        Ok(proto.int32_data.iter().map(|&v| v != 0).collect())
    }
}

/// Convert each i32 element from `int32_data` into a narrower target
/// type via `convert`, returning `ValueOverflow` on the first element
/// that does not fit. Used by the sub-32-bit decoder fallback paths
/// (INT8, UINT8, INT16, UINT16) where ONNX stores the narrower values
/// inside the wider `int32_data` field.
fn checked_narrow_i32<T>(
    path: &Path,
    data: &[i32],
    dtype: &'static str,
    convert: impl Fn(i32) -> Option<T>,
) -> Result<Vec<T>, LoadError> {
    data.iter()
        .map(|&v| {
            convert(v).ok_or_else(|| LoadError::ValueOverflow {
                path: path.to_path_buf(),
                dtype,
                value: v as i64,
            })
        })
        .collect()
}

/// Reject a `raw_data` buffer whose length is not a multiple of the
/// dtype's element size. This catches truncated or misaligned buffers
/// before they silently parse as a shorter-but-valid sequence.
fn check_raw_alignment(
    path: &Path,
    raw: &[u8],
    elem_size: usize,
    dtype: &'static str,
) -> Result<(), LoadError> {
    if !raw.len().is_multiple_of(elem_size) {
        return Err(LoadError::UnalignedRawData {
            path: path.to_path_buf(),
            len: raw.len(),
            elem_size,
            dtype,
        });
    }
    Ok(())
}
