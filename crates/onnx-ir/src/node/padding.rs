//! # Padding Configuration Utilities
//!
//! Padding configuration types for 1D, 2D, and 3D operations.
//!
//! Provides `PaddingConfig1d`, `PaddingConfig2d`, `PaddingConfig3d` enums and helper
//! functions to convert ONNX padding arrays.
//!
//! [`forward_time_same_blocker`] encodes what burn's `PaddingConfig::Same` can serve, which is
//! burn knowledge in a crate that otherwise mirrors ONNX. That is deliberate: `NodeCodegen`
//! returns no `Result`, so burn-onnx cannot reject anything without panicking, and rejecting
//! here is the only way to give the user a `ProcessError` naming the node. Treat it as an
//! exception forced by the missing error channel, not as licence to move more burn knowledge in.

use std::fmt;

use crate::ir::{ArgType, AttributeValue, RawNode};
use crate::processor::ProcessError;

/// ONNX auto_pad attribute value.
///
/// Specifies how padding should be computed automatically.
/// When set to anything other than `NotSet`, the `pads` attribute is ignored.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AutoPad {
    /// Use explicit `pads` attribute (default).
    #[default]
    NotSet,
    /// Pad so output_size = ceil(input_size / stride). Extra padding at end.
    SameUpper,
    /// Pad so output_size = ceil(input_size / stride). Extra padding at start.
    SameLower,
    /// No padding (equivalent to all-zero pads).
    Valid,
}

impl AutoPad {
    /// Parse an ONNX auto_pad string attribute.
    pub fn parse(s: &str) -> Result<Self, ProcessError> {
        match s {
            "NOTSET" => Ok(AutoPad::NotSet),
            "SAME_UPPER" => Ok(AutoPad::SameUpper),
            "SAME_LOWER" => Ok(AutoPad::SameLower),
            "VALID" => Ok(AutoPad::Valid),
            _ => Err(ProcessError::InvalidAttribute {
                name: "auto_pad".to_string(),
                reason: format!("Unknown auto_pad value: {s}"),
            }),
        }
    }
}

impl fmt::Display for AutoPad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutoPad::NotSet => write!(f, "NOTSET"),
            AutoPad::SameUpper => write!(f, "SAME_UPPER"),
            AutoPad::SameLower => write!(f, "SAME_LOWER"),
            AutoPad::Valid => write!(f, "VALID"),
        }
    }
}

/// Spatial dimensions of a convolution or pooling input, when all of them are known.
///
/// Only the dimensions after N and C are inspected: a dynamic batch size does not stop the
/// padding from being computed at import time.
pub fn static_spatial_dims(ty: &ArgType) -> Option<Vec<usize>> {
    ty.static_shape()?.get(2..)?.iter().copied().collect()
}

/// Why burn cannot defer a `SAME_UPPER`/`SAME_LOWER` padding to forward time.
///
/// Each variant names a property of burn's `calculate_same_padding`, verified against the pinned
/// burn revision. `Display` renders the sentence that goes into the user-facing `ProcessError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameBlocker {
    /// ONNX puts the extra pad of an odd total at the start, burn puts it at the end.
    SameLower,
    /// burn's forward-time padding takes no dilation, so the effective kernel would be wrong.
    Dilated,
    /// burn's 3D padding is symmetric-only, so an odd total has nowhere to go.
    ThreeSpatialDims,
}

impl fmt::Display for SameBlocker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SameBlocker::SameLower => write!(
                f,
                "SAME_LOWER puts the extra pad at the start of the dimension, while padding \
                 computed at forward time puts it at the end"
            ),
            SameBlocker::Dilated => {
                write!(
                    f,
                    "padding computed at forward time does not account for dilation"
                )
            }
            SameBlocker::ThreeSpatialDims => write!(
                f,
                "padding computed at forward time is symmetric-only over 3 spatial dimensions"
            ),
        }
    }
}

/// Why burn cannot defer this `SAME_UPPER`/`SAME_LOWER` padding to forward time, if it cannot.
///
/// `PaddingConfig::Same` derives the pads from the real input size during `forward`, which is the
/// only option left when the spatial dimensions are unknown at import time. It reproduces ONNX
/// for `SAME_UPPER` over one or two spatial dimensions without dilation.
///
/// Other cases can coincide with ONNX without being listed as safe here: an even total makes
/// SAME_LOWER and SAME_UPPER identical, and a stride-1 odd kernel is always symmetric even in 3D.
/// This check stays conservative rather than reasoning about pad parity, so it rejects them.
///
/// The answer assumes the burn module routes padding through `calculate_padding_1d_pair` or
/// `calculate_padding_2d_pairs`, which keep the two sides separate. Every module burn-onnx
/// targets today does; `DeformConv2d` does not, and would need its own check.
///
/// onnx-ir turns a blocker into a `ProcessError`; burn-onnx codegen asserts there is none before
/// emitting `Same`. Both consult this function so the rule is stated once, but each gathers its
/// arguments separately, so every op reaching a `resolve_auto_pad_*` must also call
/// [`validate_auto_pad`] for the two to agree.
pub fn forward_time_same_blocker(
    auto_pad: &AutoPad,
    dilated: bool,
    spatial_rank: usize,
) -> Option<SameBlocker> {
    if *auto_pad == AutoPad::SameLower {
        Some(SameBlocker::SameLower)
    } else if dilated {
        Some(SameBlocker::Dilated)
    } else if spatial_rank > 2 {
        Some(SameBlocker::ThreeSpatialDims)
    } else {
        None
    }
}

/// Reject a `SAME_UPPER`/`SAME_LOWER` auto_pad whose input spatial dimensions are unknown and
/// that burn cannot resolve at forward time either.
///
/// Known spatial dimensions let the pads be computed during import, which covers every mode over
/// one or two spatial dimensions. 3D still requires the resulting pads to come out symmetric;
/// `PaddingConfig3d::to_tokens` in burn-onnx rejects the rest.
pub(crate) fn validate_auto_pad(node: &RawNode) -> Result<(), ProcessError> {
    let auto_pad = match node.attrs.get("auto_pad") {
        None => return Ok(()),
        Some(AttributeValue::String(value)) => AutoPad::parse(value)?,
        Some(other) => {
            return Err(ProcessError::InvalidAttribute {
                name: "auto_pad".to_string(),
                reason: format!("expected a string, got {other:?}"),
            });
        }
    };

    let input = &node.inputs[0].ty;
    if !matches!(auto_pad, AutoPad::SameUpper | AutoPad::SameLower)
        || static_spatial_dims(input).is_some()
    {
        return Ok(());
    }

    let dilated = match node.attrs.get("dilations") {
        None => false,
        Some(AttributeValue::Int64s(dilations)) => dilations.iter().any(|&d| d != 1),
        Some(other) => {
            return Err(ProcessError::InvalidAttribute {
                name: "dilations".to_string(),
                reason: format!("expected a list of ints, got {other:?}"),
            });
        }
    };

    match forward_time_same_blocker(&auto_pad, dilated, input.rank().saturating_sub(2)) {
        None => Ok(()),
        Some(blocker) => Err(ProcessError::InvalidAttribute {
            name: "auto_pad".to_string(),
            reason: format!(
                "{auto_pad} needs the input spatial dimensions, but they are dynamic, and the \
                 padding cannot be deferred to forward time either: {blocker}. Re-export the \
                 model with a static input shape, or with explicit pads."
            ),
        }),
    }
}

/// Padding configuration for 1D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig1d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for left and right sides
    /// Format: (left, right)
    /// For symmetric padding, use the same value for both (e.g., `Explicit(1, 1)`).
    Explicit(usize, usize),
}

impl fmt::Display for PaddingConfig1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig1d::Valid => write!(f, "Valid"),
            PaddingConfig1d::Explicit(left, right) => write!(f, "Explicit({left}, {right})"),
        }
    }
}

impl PaddingConfig1d {
    /// Returns true if this padding configuration is asymmetric (left != right)
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig1d::Explicit(left, right) => left != right,
            _ => false,
        }
    }

    /// Returns the padding values as (left, right) tuple
    pub fn as_tuple(&self) -> (usize, usize) {
        match self {
            PaddingConfig1d::Valid => (0, 0),
            PaddingConfig1d::Explicit(left, right) => (*left, *right),
        }
    }
}

/// Calculate the padding configuration for a 1D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [left, right]
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_1d(pads: &[i64]) -> PaddingConfig1d {
    let [left, right] = [pads[0], pads[1]];

    if left < 0 || right < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 {
        PaddingConfig1d::Valid
    } else {
        PaddingConfig1d::Explicit(left as usize, right as usize)
    }
}

/// Padding configuration for 2D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig2d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for each side
    /// Format: (top, left, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize),
}

impl fmt::Display for PaddingConfig2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig2d::Valid => write!(f, "Valid"),
            PaddingConfig2d::Explicit(top, left, bottom, right) => {
                write!(f, "Explicit({top}, {left}, {bottom}, {right})")
            }
        }
    }
}

impl PaddingConfig2d {
    /// Returns true if this padding configuration is asymmetric (top != bottom or left != right)
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig2d::Explicit(top, left, bottom, right) => top != bottom || left != right,
            _ => false,
        }
    }

    /// Returns the padding values as (top, left, bottom, right) tuple
    pub fn as_tuple(&self) -> (usize, usize, usize, usize) {
        match self {
            PaddingConfig2d::Valid => (0, 0, 0, 0),
            PaddingConfig2d::Explicit(top, left, bottom, right) => (*top, *left, *bottom, *right),
        }
    }
}

/// Calculate the padding configuration for a 2D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [top, left, bottom, right] (ONNX format)
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_2d(pads: &[i64]) -> PaddingConfig2d {
    let [top, left, bottom, right] = [pads[0], pads[1], pads[2], pads[3]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 {
        PaddingConfig2d::Valid
    } else {
        PaddingConfig2d::Explicit(top as usize, left as usize, bottom as usize, right as usize)
    }
}

/// Padding configuration for 3D operations such as convolution
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PaddingConfig3d {
    /// No padding (valid padding)
    #[default]
    Valid,
    /// Explicit padding with values for each side
    /// Format: (front, top, left, back, bottom, right)
    /// For symmetric padding, use matching values (e.g., `Explicit(1, 1, 1, 1, 1, 1)`).
    Explicit(usize, usize, usize, usize, usize, usize),
}

impl fmt::Display for PaddingConfig3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingConfig3d::Valid => write!(f, "Valid"),
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                write!(
                    f,
                    "Explicit({front}, {top}, {left}, {back}, {bottom}, {right})"
                )
            }
        }
    }
}

impl PaddingConfig3d {
    /// Returns true if this padding configuration is asymmetric
    pub fn is_asymmetric(&self) -> bool {
        match self {
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                front != back || top != bottom || left != right
            }
            _ => false,
        }
    }

    /// Returns the padding values as (front, top, left, back, bottom, right) tuple
    pub fn as_tuple(&self) -> (usize, usize, usize, usize, usize, usize) {
        match self {
            PaddingConfig3d::Valid => (0, 0, 0, 0, 0, 0),
            PaddingConfig3d::Explicit(front, top, left, back, bottom, right) => {
                (*front, *top, *left, *back, *bottom, *right)
            }
        }
    }
}

/// Calculate the padding configuration for a 3D operations such as Convolution and Pooling.
///
/// # Arguments
///
/// * `pads` - The padding values [front, top, left, back, bottom, right] (ONNX format)
///
/// # Panics
///
/// * If the padding is negative
///
/// # Returns
///
/// * The padding configuration (Valid or Explicit)
///
/// # Remarks
///
/// This function is used when the padding is specified as a list of integers,
/// and not used when the padding is specified as a string, e.g. "SAME_UPPER".
pub(crate) fn padding_config_3d(pads: &[i64]) -> PaddingConfig3d {
    let [front, top, left, back, bottom, right] =
        [pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]];

    if left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0 || back < 0 {
        panic!("Negative pad values are not supported");
    } else if left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && back == 0 {
        PaddingConfig3d::Valid
    } else {
        PaddingConfig3d::Explicit(
            front as usize,
            top as usize,
            left as usize,
            back as usize,
            bottom as usize,
            right as usize,
        )
    }
}

/// A conv weight's shape: from a fully known static shape, or from its constant value,
/// which is only read (and copied) when the type does not carry the shape.
pub(crate) fn known_weight_shape(weight: &crate::ir::Argument) -> Option<Vec<usize>> {
    weight
        .ty
        .static_shape_known()
        .or_else(|| weight.value().map(|data| data.shape.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    /// Conv node whose input spatial dimensions are `spatial` (`None` = dynamic). The spatial
    /// rank, which decides the 3D blocker, follows from how many are given. The node type is
    /// always Conv2d even for other ranks; `validate_auto_pad` does not inspect it.
    fn conv_node(auto_pad: &str, spatial: &[Option<usize>], dilations: Vec<i64>) -> RawNode {
        let mut static_shape = vec![None, Some(3)];
        static_shape.extend_from_slice(spatial);
        let rank = static_shape.len();
        TestNodeBuilder::new(NodeType::Conv2d, "test_conv")
            .add_input(
                "data",
                ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank,
                    static_shape: Some(static_shape),
                }),
            )
            .output_tensor_f32("output", rank, None)
            .attr_string("auto_pad", auto_pad)
            .attr_ints("dilations", dilations)
            .build()
    }

    #[test]
    fn test_static_spatial_dims_ignores_dynamic_batch() {
        let node = conv_node("SAME_UPPER", &[Some(7), Some(9)], vec![1, 1]);
        assert_eq!(static_spatial_dims(&node.inputs[0].ty), Some(vec![7, 9]));
    }

    #[test]
    fn test_static_spatial_dims_dynamic() {
        let node = conv_node("SAME_UPPER", &[None, None], vec![1, 1]);
        assert_eq!(static_spatial_dims(&node.inputs[0].ty), None);
    }

    #[test]
    fn test_validate_auto_pad_accepts_static_spatial() {
        // Over 1 or 2 spatial dimensions every mode is fine once the dimensions are known:
        // the pads are computed during import.
        for auto_pad in ["SAME_UPPER", "SAME_LOWER"] {
            let node = conv_node(auto_pad, &[Some(7), Some(7)], vec![2, 2]);
            assert!(validate_auto_pad(&node).is_ok());
        }
    }

    #[test]
    fn test_validate_auto_pad_accepts_same_upper_dynamic() {
        let node = conv_node("SAME_UPPER", &[None, None], vec![1, 1]);
        assert!(validate_auto_pad(&node).is_ok());
    }

    #[test]
    fn test_validate_auto_pad_rejects_same_lower_dynamic() {
        let node = conv_node("SAME_LOWER", &[None, None], vec![1, 1]);
        let err = validate_auto_pad(&node).unwrap_err().to_string();
        assert!(err.contains(&SameBlocker::SameLower.to_string()), "{err}");
    }

    #[test]
    fn test_validate_auto_pad_rejects_dilation_dynamic() {
        let node = conv_node("SAME_UPPER", &[None, None], vec![2, 2]);
        let err = validate_auto_pad(&node).unwrap_err().to_string();
        assert!(err.contains(&SameBlocker::Dilated.to_string()), "{err}");
    }

    #[test]
    fn test_validate_auto_pad_rejects_3d_dynamic() {
        let node = conv_node("SAME_UPPER", &[None, None, None], vec![1, 1, 1]);
        let err = validate_auto_pad(&node).unwrap_err().to_string();
        assert!(
            err.contains(&SameBlocker::ThreeSpatialDims.to_string()),
            "{err}"
        );
    }

    #[test]
    fn test_forward_time_same_blocker_precedence() {
        // SAME_LOWER is reported before dilation, and dilation before the 3D limit, so the
        // message names the first thing the user has to change.
        assert_eq!(
            forward_time_same_blocker(&AutoPad::SameLower, true, 3),
            Some(SameBlocker::SameLower)
        );
        assert_eq!(
            forward_time_same_blocker(&AutoPad::SameUpper, true, 3),
            Some(SameBlocker::Dilated)
        );
        assert_eq!(
            forward_time_same_blocker(&AutoPad::SameUpper, false, 3),
            Some(SameBlocker::ThreeSpatialDims)
        );
        assert_eq!(
            forward_time_same_blocker(&AutoPad::SameUpper, false, 2),
            None
        );
    }

    #[test]
    fn test_validate_auto_pad_rejects_wrong_attribute_types() {
        let mut node = conv_node("SAME_UPPER", &[None, None], vec![1, 1]);
        node.attrs
            .insert("auto_pad".to_string(), AttributeValue::Int64(1));
        let err = validate_auto_pad(&node).unwrap_err().to_string();
        assert!(err.contains("expected a string"), "{err}");

        let mut node = conv_node("SAME_UPPER", &[None, None], vec![1, 1]);
        node.attrs
            .insert("dilations".to_string(), AttributeValue::Int64(1));
        let err = validate_auto_pad(&node).unwrap_err().to_string();
        assert!(err.contains("expected a list of ints"), "{err}");
    }

    #[test]
    fn test_validate_auto_pad_ignores_other_modes() {
        for auto_pad in ["NOTSET", "VALID"] {
            let node = conv_node(auto_pad, &[None, None], vec![1, 1]);
            assert!(validate_auto_pad(&node).is_ok());
        }
    }

    // AutoPad tests
    #[test]
    fn test_auto_pad_parse() {
        assert_eq!(AutoPad::parse("NOTSET").unwrap(), AutoPad::NotSet);
        assert_eq!(AutoPad::parse("SAME_UPPER").unwrap(), AutoPad::SameUpper);
        assert_eq!(AutoPad::parse("SAME_LOWER").unwrap(), AutoPad::SameLower);
        assert_eq!(AutoPad::parse("VALID").unwrap(), AutoPad::Valid);
        assert!(AutoPad::parse("INVALID").is_err());
    }

    #[test]
    fn test_auto_pad_display() {
        assert_eq!(AutoPad::NotSet.to_string(), "NOTSET");
        assert_eq!(AutoPad::SameUpper.to_string(), "SAME_UPPER");
        assert_eq!(AutoPad::SameLower.to_string(), "SAME_LOWER");
        assert_eq!(AutoPad::Valid.to_string(), "VALID");
    }

    // 1D padding tests
    #[test]
    fn test_padding_config_1d_valid() {
        let pads = vec![0, 0];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Valid));
    }

    #[test]
    fn test_padding_config_1d_explicit_symmetric() {
        let pads = vec![2, 2];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Explicit(2, 2)));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 2));
    }

    #[test]
    fn test_padding_config_1d_explicit_asymmetric() {
        let pads = vec![1, 2];
        let config = padding_config_1d(&pads);
        assert!(matches!(config, PaddingConfig1d::Explicit(1, 2)));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2));
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_1d_negative() {
        let pads = vec![-1, -1];
        let _ = padding_config_1d(&pads);
    }

    // 2D padding tests
    #[test]
    fn test_padding_config_2d_valid() {
        let pads = vec![0, 0, 0, 0];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Valid));
        assert!(!config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_explicit_symmetric() {
        let pads = vec![2, 2, 2, 2];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(2, 2, 2, 2)));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 2, 2, 2));
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric() {
        // pads = [top, left, bottom, right]
        let pads = vec![1, 2, 3, 4];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(1, 2, 3, 4)));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2, 3, 4));
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric_top_bottom() {
        // top != bottom but left == right
        let pads = vec![1, 2, 3, 2];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(1, 2, 3, 2)));
        assert!(config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_2d_explicit_asymmetric_left_right() {
        // left != right but top == bottom
        let pads = vec![2, 1, 2, 3];
        let config = padding_config_2d(&pads);
        assert!(matches!(config, PaddingConfig2d::Explicit(2, 1, 2, 3)));
        assert!(config.is_asymmetric());
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_2d_negative() {
        let pads = vec![-1, -1, -1, -1];
        let _ = padding_config_2d(&pads);
    }

    // 3D padding tests
    #[test]
    fn test_padding_config_3d_valid() {
        let pads = vec![0, 0, 0, 0, 0, 0];
        let config = padding_config_3d(&pads);
        assert!(matches!(config, PaddingConfig3d::Valid));
        assert!(!config.is_asymmetric());
    }

    #[test]
    fn test_padding_config_3d_explicit_symmetric() {
        let pads = vec![2, 3, 1, 2, 3, 1];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(2, 3, 1, 2, 3, 1)
        ));
        assert!(!config.is_asymmetric());
        assert_eq!(config.as_tuple(), (2, 3, 1, 2, 3, 1));
    }

    #[test]
    fn test_padding_config_3d_explicit_asymmetric() {
        // pads = [front, top, left, back, bottom, right]
        let pads = vec![1, 2, 3, 4, 5, 6];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(1, 2, 3, 4, 5, 6)
        ));
        assert!(config.is_asymmetric());
        assert_eq!(config.as_tuple(), (1, 2, 3, 4, 5, 6));
    }

    #[test]
    fn test_padding_config_3d_explicit_asymmetric_partial() {
        // Only front != back
        let pads = vec![1, 3, 1, 2, 3, 1];
        let config = padding_config_3d(&pads);
        assert!(matches!(
            config,
            PaddingConfig3d::Explicit(1, 3, 1, 2, 3, 1)
        ));
        assert!(config.is_asymmetric());
    }

    #[test]
    #[should_panic(expected = "Negative pad values are not supported")]
    fn test_padding_config_3d_negative() {
        let pads = vec![-1, -1, -1, -1, -1, -1];
        let _ = padding_config_3d(&pads);
    }
}
