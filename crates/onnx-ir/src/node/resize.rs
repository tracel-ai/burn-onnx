//! # Resize
//!
//! Resizes input tensor using various interpolation methods.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Resize.html>
//!
//! ## Opset Versions
//! - **Opset 10**: Initial version with scales and sizes inputs.
//! - **Opset 11**: Added coordinate_transformation_mode attribute for more control over interpolation. Added support for linear mode (previously only nearest).
//! - **Opset 13**: Added cubic mode support and cubic_coeff_a attribute. Added antialias attribute for downsampling.
//! - **Opset 18**: Added keep_aspect_ratio_policy and axes attributes for selective resizing.
//! - **Opset 19**: Added antialiasing improvements and clarified coordinate transformation modes.
//!
//! **Implementation Note**: This implementation requires opset 11+ for coordinate transformation mode support. Many attributes are ignored or have restricted values (see validation in infer_types).
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::Argument;

use crate::ir::{ArgType, Node, RawNode, RuntimeInputRef};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

use std::str::FromStr;

/// Interpolation mode for resize operation
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ResizeMode {
    /// Nearest neighbor interpolation
    #[default]
    Nearest,
    /// Linear interpolation (bilinear for 2D, trilinear for 3D)
    Linear,
    /// Cubic interpolation
    Cubic,
}

impl FromStr for ResizeMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nearest" => Ok(ResizeMode::Nearest),
            "linear" => Ok(ResizeMode::Linear),
            "cubic" => Ok(ResizeMode::Cubic),
            _ => Err(format!("Unsupported resize mode: {}", s)),
        }
    }
}

/// Coordinate transformation mode for resize operation.
///
/// Determines how coordinates in the resized tensor map to coordinates in the original tensor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CoordinateTransformMode {
    /// Half-pixel coordinate transformation (default for opset 11+)
    #[default]
    HalfPixel,
    /// Align corners coordinate transformation
    AlignCorners,
    /// Asymmetric coordinate transformation (default for opset <11)
    Asymmetric,
    /// PyTorch-style half-pixel transformation
    PytorchHalfPixel,
    /// TensorFlow crop-and-resize transformation
    TfCropAndResize,
    /// TensorFlow half-pixel for nearest-neighbor
    TfHalfPixelForNn,
}

impl FromStr for CoordinateTransformMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "half_pixel" => Ok(Self::HalfPixel),
            "align_corners" => Ok(Self::AlignCorners),
            "asymmetric" => Ok(Self::Asymmetric),
            "pytorch_half_pixel" => Ok(Self::PytorchHalfPixel),
            "tf_crop_and_resize" => Ok(Self::TfCropAndResize),
            "tf_half_pixel_for_nn" => Ok(Self::TfHalfPixelForNn),
            _ => Err(format!("Unsupported coordinate transformation mode: {}", s)),
        }
    }
}

/// Nearest-neighbor rounding mode for resize operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NearestMode {
    /// Round half down (default)
    #[default]
    RoundPreferFloor,
    /// Round half up
    RoundPreferCeil,
    /// Floor rounding
    Floor,
    /// Ceil rounding
    Ceil,
}

impl FromStr for NearestMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "round_prefer_floor" => Ok(Self::RoundPreferFloor),
            "round_prefer_ceil" => Ok(Self::RoundPreferCeil),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            _ => Err(format!("Unsupported nearest mode: {}", s)),
        }
    }
}

/// Configuration for the Resize operation.
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct ResizeConfig {
    pub mode: ResizeMode,
    pub scales: Option<ResizeScales>,
    pub sizes: Option<ResizeSizes>,
    /// Coordinate transformation mode
    pub coordinate_transformation_mode: CoordinateTransformMode,
    /// Cubic coefficient for cubic interpolation (default: -0.75)
    pub cubic_coeff_a: f32,
    /// Nearest mode rounding strategy
    pub nearest_mode: NearestMode,
    /// Exclude outside weights (default: 0)
    pub exclude_outside: i32,
    /// Extrapolation value for tf_crop_and_resize mode (default: 0.0)
    pub extrapolation_value: f32,
    /// Antialias flag (default: 0) - opset 13+
    pub antialias: i32,
}

impl Default for ResizeConfig {
    fn default() -> Self {
        Self {
            mode: ResizeMode::Nearest,
            scales: None,
            sizes: None,
            coordinate_transformation_mode: CoordinateTransformMode::HalfPixel,
            cubic_coeff_a: -0.75,
            nearest_mode: NearestMode::RoundPreferFloor,
            exclude_outside: 0,
            extrapolation_value: 0.0,
            antialias: 0,
        }
    }
}

/// Represents either a static value or a runtime argument for resize scales.
#[derive(Debug, Clone)]
pub enum ResizeScales {
    /// Static scales known at compile time.
    Static(Vec<f32>),
    /// Runtime scales determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ResizeScales {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Represents either a static value or a runtime argument for resize sizes.
#[derive(Debug, Clone)]
pub enum ResizeSizes {
    /// Static sizes known at compile time.
    Static(Vec<usize>),
    /// Runtime sizes determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ResizeSizes {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Node representation for Resize operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ResizeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ResizeConfig,
}

/// Extract scales input as either static or runtime
fn extract_scales_input(node: &RawNode, input_rank: usize, idx: usize) -> Option<ResizeScales> {
    match node.inputs.get(idx) {
        Some(input) => {
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a static value (lifted constant) or constant
                    match input.value() {
                        Some(tensor_data) => {
                            let mut scales: Vec<f32> = tensor_data.to_vec().unwrap();
                            if scales.is_empty() {
                                return None;
                            }
                            assert!(scales.len() == input_rank);
                            // ignore the first two items from scales
                            // because they are the batch and channel dimensions
                            scales = scales.iter().skip(2).cloned().collect();
                            Some(ResizeScales::Static(scales))
                        }
                        None => {
                            // Runtime input - store reference instead of cloning the argument
                            Some(ResizeScales::Runtime(RuntimeInputRef::new(
                                input.name.clone(),
                                idx,
                            )))
                        }
                    }
                }
                ArgType::Shape(_) => {
                    // Shape input for scales - store reference instead of cloning the argument
                    Some(ResizeScales::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        idx,
                    )))
                }
                _ => None,
            }
        }
        None => None,
    }
}

/// Extract sizes input as either static or runtime
fn extract_sizes_input(node: &RawNode, input_rank: usize, idx: usize) -> Option<ResizeSizes> {
    match node.inputs.get(idx) {
        Some(input) => {
            // Skip optional inputs (those that were never provided)
            if input.is_optional() {
                return None;
            }

            match &input.ty {
                ArgType::Tensor(_) => {
                    // Check if it's a static value (lifted constant) or constant
                    match input.value() {
                        Some(tensor_data) => {
                            let i64_sizes: Vec<i64> = tensor_data.to_vec().unwrap();
                            let mut sizes: Vec<usize> =
                                i64_sizes.iter().map(|&x| x as usize).collect();
                            if sizes.is_empty() {
                                return None;
                            }
                            assert!(sizes.len() == input_rank);
                            // ignore the first two items from sizes
                            // because they are the batch and channel dimensions
                            sizes = sizes.iter().skip(2).cloned().collect();
                            Some(ResizeSizes::Static(sizes))
                        }
                        None => {
                            // Runtime input - store reference instead of cloning the argument
                            Some(ResizeSizes::Runtime(RuntimeInputRef::new(
                                input.name.clone(),
                                idx,
                            )))
                        }
                    }
                }
                ArgType::Shape(_rank) => {
                    // Shape input for sizes - store reference instead of cloning the argument
                    // The Shape type represents the shape of a tensor, which is exactly what we need
                    Some(ResizeSizes::Runtime(RuntimeInputRef::new(
                        input.name.clone(),
                        idx,
                    )))
                }
                _ => None,
            }
        }
        None => None,
    }
}

pub(crate) struct ResizeProcessor;

impl NodeProcessor for ResizeProcessor {
    type Config = ResizeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(1, 4),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Lift roi input (input[1]) if present and constant
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        // Lift scales input (input[2]) if present and constant
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }

        // Lift sizes input (input[3]) if present and constant
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "antialias" => {
                    if value.clone().into_i32() != 0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "antialias".to_string(),
                            reason: "antialias other than 0 is not supported".to_string(),
                        });
                    }
                }
                "axes" => {
                    return Err(ProcessError::InvalidAttribute {
                        name: "axes".to_string(),
                        reason: "custom axes attribute is not supported".to_string(),
                    });
                }
                "coordinate_transformation_mode" | "cubic_coeff_a" => {
                    // Parsed in extract_config
                }
                "exclude_outside" => {
                    if value.clone().into_i32() != 0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "exclude_outside".to_string(),
                            reason: "exclude_outside other than 0 is not supported".to_string(),
                        });
                    }
                }
                "extrapolation_value" => {
                    if value.clone().into_f32() != 0.0 {
                        return Err(ProcessError::InvalidAttribute {
                            name: "extrapolation_value".to_string(),
                            reason: "extrapolation_value other than 0.0 is not supported"
                                .to_string(),
                        });
                    }
                }
                "keep_aspect_ratio_policy" => {
                    if value.clone().into_string().to_lowercase() != "stretch" {
                        return Err(ProcessError::InvalidAttribute {
                            name: "keep_aspect_ratio_policy".to_string(),
                            reason:
                                "keep_aspect_ratio_policy other than 'stretch' is not supported"
                                    .to_string(),
                        });
                    }
                }
                "mode" | "nearest_mode" => {
                    // Parsed in extract_config
                }
                _ => {}
            }
        }

        // Opset 10: inputs are [X, scales] (no roi input)
        // Opset 11+: inputs are [X, roi, scales, sizes]
        if opset >= 11 {
            let roi: Vec<f32> = node
                .inputs
                .get(1)
                .map(|input| {
                    if let Some(tensor_data) = input.value() {
                        tensor_data.to_vec().unwrap()
                    } else {
                        vec![]
                    }
                })
                .unwrap_or_default();

            if !roi.is_empty() {
                return Err(ProcessError::Custom(
                    "Resize: roi input is not supported".to_string(),
                ));
            }
        }

        let config = self.extract_config(node, opset)?;

        // Exactly one of scales or sizes must be provided
        match (&config.scales, &config.sizes) {
            (None, None) => {
                return Err(ProcessError::Custom(
                    "Resize: either scales or sizes input is required".to_string(),
                ));
            }
            (Some(_), Some(_)) => {
                return Err(ProcessError::Custom(
                    "Resize: scales and sizes are mutually exclusive".to_string(),
                ));
            }
            _ => {}
        }

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        let mut mode: Option<ResizeMode> = None;
        // Opset 10 had no coordinate_transformation_mode (implicit asymmetric)
        let mut coordinate_transformation_mode = if opset < 11 {
            CoordinateTransformMode::Asymmetric
        } else {
            CoordinateTransformMode::HalfPixel
        };
        let mut cubic_coeff_a = -0.75f32;
        let mut nearest_mode = NearestMode::RoundPreferFloor;
        let mut exclude_outside = 0i32;
        let mut extrapolation_value = 0.0f32;
        let mut antialias = 0i32;

        let input = if let ArgType::Tensor(tensor) = &node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("input".to_string()))?
            .ty
        {
            tensor
        } else {
            return Err(ProcessError::TypeMismatch {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", node.inputs.first().unwrap().ty),
            });
        };

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "mode" => {
                    mode = Some(
                        value
                            .clone()
                            .into_string()
                            .parse::<ResizeMode>()
                            .map_err(|e| ProcessError::InvalidAttribute {
                                name: "mode".to_string(),
                                reason: format!("Failed to parse resize mode: {}", e),
                            })?,
                    )
                }
                "coordinate_transformation_mode" => {
                    coordinate_transformation_mode = value
                        .clone()
                        .into_string()
                        .parse::<CoordinateTransformMode>()
                        .map_err(|e| ProcessError::InvalidAttribute {
                            name: "coordinate_transformation_mode".to_string(),
                            reason: e,
                        })?;
                }
                "cubic_coeff_a" => {
                    cubic_coeff_a = value.clone().into_f32();
                }
                "nearest_mode" => {
                    nearest_mode =
                        value
                            .clone()
                            .into_string()
                            .parse::<NearestMode>()
                            .map_err(|e| ProcessError::InvalidAttribute {
                                name: "nearest_mode".to_string(),
                                reason: e,
                            })?;
                }
                "exclude_outside" => {
                    exclude_outside = value.clone().into_i32();
                }
                "extrapolation_value" => {
                    extrapolation_value = value.clone().into_f32();
                }
                "antialias" => {
                    antialias = value.clone().into_i32();
                }
                _ => {}
            }
        }

        // Opset 10: inputs are [X, scales] (no roi, no sizes)
        // Opset 11+: inputs are [X, roi, scales, sizes]
        let (scales_idx, sizes_idx) = if opset < 11 { (1, usize::MAX) } else { (2, 3) };

        let scales = extract_scales_input(node, input.rank, scales_idx);
        let sizes = extract_sizes_input(node, input.rank, sizes_idx);

        let mode = mode.ok_or_else(|| ProcessError::MissingAttribute("mode".to_string()))?;

        let config = ResizeConfig {
            mode,
            scales,
            sizes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            nearest_mode,
            exclude_outside,
            extrapolation_value,
            antialias,
        };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Resize(ResizeNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(
        mode: &str,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
        roi: Option<Vec<f32>>,
    ) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None) // N,C,H,W format
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", mode);

        // Add ROI input if provided
        if let Some(roi_data) = roi {
            builder = builder.input_tensor_f32_data("roi", roi_data.clone(), vec![8]);
            // For 4D input (start x, start y, end x, end y)
        } else {
            // Empty ROI still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add scales input if provided
        if let Some(scales_data) = scales {
            builder = builder.input_tensor_f32_data("scales", scales_data.clone(), vec![4]);
            // N,C,H,W scales
        } else {
            // Empty scales still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_f32("", 1, None);
        }

        // Add sizes input if provided
        if let Some(sizes_data) = sizes {
            builder = builder.input_tensor_i64_data("sizes", sizes_data.clone(), vec![4]);
            // N,C,H,W sizes
        } else {
            // Empty sizes still needs to be added as a placeholder with empty name
            builder = builder.input_tensor_i64("", 1, None);
        }

        builder
    }

    #[test]
    fn test_resize_config_with_scales() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]), // Keep N,C same, double H,W
            None,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.mode, ResizeMode::Nearest);
        match &config.scales {
            Some(ResizeScales::Static(scales)) => {
                assert_eq!(*scales, vec![2.0, 2.0]); // Only the spatial scales (H,W)
            }
            _ => panic!("Expected static scales"),
        }
        assert!(config.sizes.is_none(), "Expected no sizes");
        // Verify default attribute values
        assert_eq!(
            config.coordinate_transformation_mode,
            CoordinateTransformMode::HalfPixel
        );
        assert_eq!(config.cubic_coeff_a, -0.75);
        assert_eq!(config.nearest_mode, NearestMode::RoundPreferFloor);
        assert_eq!(config.exclude_outside, 0);
        assert_eq!(config.extrapolation_value, 0.0);
        assert_eq!(config.antialias, 0);
    }

    #[test]
    fn test_resize_config_with_sizes() {
        let node = create_test_node(
            "linear",
            None,
            Some(vec![1, 3, 224, 224]), // Fixed output size
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.mode, ResizeMode::Linear);
        assert!(config.scales.is_none(), "Expected no scales");
        match &config.sizes {
            Some(ResizeSizes::Static(sizes)) => {
                assert_eq!(*sizes, vec![224, 224]); // Only the spatial sizes (H,W)
            }
            _ => panic!("Expected static sizes"),
        }
    }

    #[test]
    fn test_resize_config_with_roi() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            None,
            Some(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), // ROI values
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_scales_or_sizes() {
        let node = create_test_node("nearest", None, None, None).build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let _config = processor.extract_config(&node, 16).unwrap();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_config_no_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.clear(); // Remove all attributes including mode
        let node = node;
        let processor = ResizeProcessor;
        let _prefs = OutputPreferences::new();
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::MissingAttribute(_))));
    }

    #[test]
    fn test_resize_invalid_coordinate_transformation_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.insert(
            "coordinate_transformation_mode".to_string(),
            crate::ir::AttributeValue::String("invalid_mode".to_string()),
        );
        let processor = ResizeProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_resize_invalid_nearest_mode() {
        let mut node = create_test_node("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None, None)
            .build_with_graph_data(16);
        node.attrs.insert(
            "nearest_mode".to_string(),
            crate::ir::AttributeValue::String("bad_mode".to_string()),
        );
        let processor = ResizeProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_resize_scales_and_sizes_both_provided() {
        let node = create_test_node(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            Some(vec![1, 3, 224, 224]),
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = ResizeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_resize_coordinate_transformation_mode_opset10_default() {
        // For opset 10, only inputs are [X, scales] (no roi, no sizes)
        let node = TestNodeBuilder::new(NodeType::Resize, "test_resize")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_string("mode", "nearest")
            .input_tensor_f32_data("scales", vec![1.0, 1.0, 2.0, 2.0], vec![4])
            .build_with_graph_data(10);
        let processor = ResizeProcessor;
        let config = processor.extract_config(&node, 10).unwrap();
        assert_eq!(
            config.coordinate_transformation_mode,
            CoordinateTransformMode::Asymmetric
        );
    }
}
