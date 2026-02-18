//! # Reshape
//!
//! Reshapes the input tensor to a new shape specified by the shape input.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Reshape.html>
//!
//! ## Special Features
//! - The `shape` input can contain special values:
//!   - `-1`: At most one dimension can be -1, which will be inferred from the tensor size
//!     and remaining dimensions
//!   - `0`: When allowzero=0 (default), copies the corresponding dimension from input tensor.
//!     When allowzero=1, sets the dimension to zero explicitly
//!   - Empty shape: Converts tensor to a scalar
//!
//! **NOTE**: The `allowzero` attribute (opset 14+) IS now validated in infer_types (lines 346-363).
//! When allowzero=1, the implementation correctly checks that shape cannot contain both 0 and -1.
//! However, the actual reshape logic respecting allowzero=1 behavior needs verification in codegen.
//!
//! ## Opset Versions
//! - **Opset 1-4**: Used 'shape' attribute.
//! - **Opset 5**: Changed shape from attribute to input, enabling dynamic reshaping.
//! - **Opset 13**: Added support for more data types including bfloat16.
//! - **Opset 14**: Added 'allowzero' attribute to control zero-dimension handling.
//! - **Opset 19**: Clarified behavior and type constraints.
//! - **Opset 21**: Added support for 8-bit integer types (int4, uint4).
//!
//! This implementation supports all opset versions. For opset < 5, shape is read from the
//! `shape` attribute. For opset 5+, shape is read from the second input.

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, RuntimeInputRef, TensorDataExt, TensorType};
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError,
};

/// Configuration for the Reshape operation.
#[derive(Debug, Clone, new)]
pub struct ReshapeConfig {
    pub shape: ReshapeInput,
}

/// Represents either a static value or a runtime argument for reshape shape.
#[derive(Debug, Clone)]
pub enum ReshapeInput {
    /// Static shape known at compile time.
    Static(Vec<i64>),
    /// Runtime shape determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ReshapeInput {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Node representation for Reshape operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ReshapeNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ReshapeConfig,
}

/// Extract relevant information from input argument
struct InputInfo {
    dtype: crate::ir::DType,
    is_shape: bool,
    shape_size: Option<usize>,
}

fn extract_input_info(input: &Argument) -> InputInfo {
    match &input.ty {
        ArgType::Tensor(tensor) => InputInfo {
            dtype: tensor.dtype,
            is_shape: false,
            shape_size: None,
        },
        ArgType::Shape(size) => InputInfo {
            dtype: crate::ir::DType::I64,
            is_shape: true,
            shape_size: Some(*size),
        },
        ArgType::ScalarTensor(dtype) | ArgType::ScalarNative(dtype) => {
            // Scalar can be used as input when reshaping to/from rank 0
            InputInfo {
                dtype: *dtype,
                is_shape: false,
                shape_size: None,
            }
        }
    }
}

/// Determine the output type based on input and output characteristics
fn determine_output_type(
    input: &Argument,
    input_info: &InputInfo,
    output_rank: usize,
    static_shape: Option<Vec<Option<usize>>>,
    node: &RawNode,
) -> ArgType {
    // Case 1: Scalar output (rank 0)
    if output_rank == 0 {
        return ArgType::ScalarNative(input_info.dtype);
    }

    // Case 2: Scalar input reshaped to [1] or [-1] - keep as scalar
    // This avoids unnecessary scalar -> tensor -> scalar conversions
    if matches!(input.ty, ArgType::ScalarNative(_))
        && output_rank == 1
        && let Some(shape_values) = get_static_shape(node)
    {
        // Shape is [-1] or [1] - effectively a single element, keep as scalar
        if shape_values.len() == 1 && (shape_values[0] == -1 || shape_values[0] == 1) {
            return ArgType::ScalarNative(input_info.dtype);
        }
    }

    // Case 3: Shape input -> Shape output (optimization)
    if input_info.is_shape && output_rank == 1 && input_info.dtype == crate::ir::DType::I64 {
        let output_size =
            calculate_shape_output_size(input_info.shape_size.unwrap_or(1), node, &static_shape);

        return ArgType::Shape(output_size);
    }

    // Case 4: Regular tensor output
    ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape,
        dtype: input_info.dtype,
    })
}

/// Calculate the output size for Shape type outputs
fn calculate_shape_output_size(
    input_size: usize,
    node: &RawNode,
    static_shape: &Option<Vec<Option<usize>>>,
) -> usize {
    // Try to get size from static reshape parameter
    if let Some(shape_values) = get_static_shape(node)
        && shape_values.len() == 1
    {
        return match shape_values[0] {
            -1 => input_size, // Infer dimension
            n if n > 0 => n as usize,
            _ => 1, // Invalid value, default to 1
        };
    }

    // Try to get size from output's static shape
    if let Some(shape) = static_shape
        && shape.len() == 1
        && let Some(dim) = shape[0]
    {
        return dim;
    }

    // Default: preserve input size
    input_size
}

/// Infer output rank for reshape operation from available information
fn infer_reshape_output_rank(node: &RawNode) -> usize {
    // Try sources in order of preference

    // 1. Static shape from constant shape input
    if let Some(shape) = get_static_shape(node) {
        return shape.len();
    }

    // 2. Dynamic shape from shape input type
    if let Some(rank) = get_rank_from_shape_input(node) {
        return rank;
    }

    // 3. Output's static shape if available
    if let Some(rank) = get_rank_from_output(node) {
        return rank;
    }

    // No rank information available
    panic!(
        "Reshape node {} has dynamic shape with no rank information available. \
         Cannot determine output rank.",
        node.name
    )
}

/// Get rank from shape input if available
fn get_rank_from_shape_input(node: &RawNode) -> Option<usize> {
    if node.inputs.len() != 2 {
        return None;
    }

    match &node.inputs[1].ty {
        ArgType::Shape(rank) => Some(*rank),
        ArgType::Tensor(tensor) => tensor
            .static_shape
            .as_ref()
            .filter(|dims| !dims.is_empty())
            .and_then(|dims| dims[0]),
        _ => None,
    }
}

/// Get rank from output tensor if available
fn get_rank_from_output(node: &RawNode) -> Option<usize> {
    match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => Some(tensor.rank),
        ArgType::ScalarNative(_) => Some(0),
        _ => None,
    }
}

/// Extract static shape from reshape node if available
fn get_static_shape(node: &RawNode) -> Option<Vec<i64>> {
    // Check shape input (opset 5+)
    if node.inputs.len() >= 2
        && let Some(value) = node.inputs[1].value()
    {
        return value.to_i64_vec().ok();
    }

    // Check shape attribute (opset 1-4)
    if let Some(attr) = node.attrs.get("shape") {
        return Some(attr.clone().into_i64s());
    }

    None
}

/// Node processor for Reshape operation
pub(crate) struct ReshapeProcessor;

impl NodeProcessor for ReshapeProcessor {
    type Config = ReshapeConfig;

    fn is_noop(&self, node: &RawNode) -> bool {
        // Reshape is a no-op when both input and output are Scalar
        if matches!(node.inputs[0].ty, ArgType::ScalarNative(_))
            && matches!(node.outputs[0].ty, ArgType::ScalarNative(_))
        {
            return true;
        }

        // Reshape is a no-op when input and output have identical static shapes
        if let (ArgType::Tensor(input_t), ArgType::Tensor(output_t)) =
            (&node.inputs[0].ty, &node.outputs[0].ty)
            && let (Some(in_shape), Some(out_shape)) =
                (&input_t.static_shape, &output_t.static_shape)
        {
            return in_shape == out_shape;
        }

        false
    }

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Only lift shape input (input[1]) if it has a static value
        // If it's a runtime argument (no value), it should remain in the graph
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn input_preferences(
        &self,
        node: &RawNode,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() < 2 {
            return Ok(None);
        }

        // Prefer Shape type for shape input (second input)
        Ok(Some(
            InputPreferences::new().add(&node.inputs[1].name, ArgPreference::Shape),
        ))
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing test coverage for allowzero=1 behavior
        // While allowzero attribute is validated (lines 346-363), there's no test that verifies
        // the actual reshape behavior when allowzero=1 and shape contains 0.
        // According to spec: with allowzero=1, a 0 in shape means "set dimension to 0",
        // not "copy from input". Add test: reshape_allowzero_explicit_zero

        // TODO: Missing test coverage for invalid shape values
        // Shape can contain negative values other than -1 (e.g., -2, -3). These should be rejected.
        // Add test: reshape_invalid_negative_value

        // TODO: Missing test coverage for more than one -1 in shape
        // Spec allows "at most one dimension" to be -1. Multiple -1s are invalid.
        // This is validated (line 338-342) but no test. Add test: reshape_multiple_infer_dim

        // TODO: Missing test coverage for incompatible total element count
        // When reshape shape specifies total elements != input total elements (and no -1 to infer),
        // this should fail. Add test: reshape_incompatible_size

        // Validate shape input type when provided as input (opset 5+)
        if node.inputs.len() >= 2 {
            match &node.inputs[1].ty {
                ArgType::Tensor(t) => {
                    // Shape tensor must be 1D and int64 dtype
                    if t.rank != 1 {
                        return Err(ProcessError::Custom(format!(
                            "Reshape: shape tensor must be 1D, got rank {}",
                            t.rank
                        )));
                    }
                    if t.dtype != crate::ir::DType::I64 {
                        return Err(ProcessError::TypeMismatch {
                            expected: "Shape tensor with dtype I64".to_string(),
                            actual: format!("Shape tensor with dtype {:?}", t.dtype),
                        });
                    }
                }
                ArgType::Shape(_) => {
                    // Shape type is valid
                }
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Tensor or Shape for shape input".to_string(),
                        actual: format!("{:?}", node.inputs[1].ty),
                    });
                }
            }
        }

        // Validate static shape values if available (from input or attribute)
        if let Some(shape_values) = get_static_shape(node) {
            // Count how many -1 values we have (at most one is allowed)
            let neg_one_count = shape_values.iter().filter(|&&v| v == -1).count();
            if neg_one_count > 1 {
                return Err(ProcessError::Custom(
                    "Reshape: shape can contain at most one -1 value".to_string(),
                ));
            }

            // If allowzero attribute is set, validate that we don't have both 0 and -1
            let mut allowzero = 0i64;
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "allowzero" {
                    allowzero = value.clone().into_i64();
                    break;
                }
            }

            if allowzero == 1 {
                let has_zero = shape_values.contains(&0);
                let has_neg_one = shape_values.contains(&(-1));
                if has_zero && has_neg_one {
                    return Err(ProcessError::InvalidAttribute {
                        name: "allowzero".to_string(),
                        reason: "When allowzero=1, shape cannot contain both 0 and -1".to_string(),
                    });
                }
            }
        }

        // Extract input information
        let input_info = extract_input_info(&node.inputs[0]);

        // Determine output rank
        let output_rank = infer_reshape_output_rank(node);

        // Check allowzero attribute for static_shape computation
        let allowzero = node
            .attrs
            .get("allowzero")
            .map(|v| v.clone().into_i64())
            .unwrap_or(0);

        // Compute static_shape from shape input values
        let static_shape = if let Some(shape_values) = get_static_shape(node) {
            let input_static = match &node.inputs[0].ty {
                ArgType::Tensor(t) => t.static_shape.as_ref(),
                _ => None,
            };

            let mut dims: Vec<Option<usize>> = shape_values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if v > 0 {
                        Some(v as usize)
                    } else if v == 0 {
                        if allowzero == 1 {
                            // allowzero=1: 0 means literal zero dimension
                            Some(0)
                        } else {
                            // allowzero=0 (default): copy from input at same position
                            input_static.and_then(|s| s.get(i).copied().flatten())
                        }
                    } else {
                        // v == -1: infer later
                        None
                    }
                })
                .collect();

            // Try to resolve -1 dimension
            if shape_values.contains(&-1) {
                let input_total = input_static
                    .and_then(|s| s.iter().try_fold(1usize, |acc, dim| dim.map(|d| acc * d)));
                if let Some(total) = input_total {
                    let known_product: Option<usize> = dims
                        .iter()
                        .filter(|d| d.is_some())
                        .try_fold(1usize, |acc, d| d.map(|v| acc * v));
                    if let Some(product) = known_product
                        && product > 0
                        && total % product == 0
                    {
                        let inferred = total / product;
                        for (i, v) in shape_values.iter().enumerate() {
                            if *v == -1 {
                                dims[i] = Some(inferred);
                            }
                        }
                    }
                }
            }

            Some(dims)
        } else {
            // Fall back to output's existing static_shape
            match &node.outputs[0].ty {
                ArgType::Tensor(t) => t.static_shape.clone(),
                _ => None,
            }
        };

        // Set output type
        node.outputs[0].ty = determine_output_type(
            &node.inputs[0],
            &input_info,
            output_rank,
            static_shape,
            node,
        );

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Check for shape attribute (opset 1-4)
        if node.inputs.len() < 2 {
            if let Some(attr) = node.attrs.get("shape") {
                let shape = attr.clone().into_i64s();
                return Ok(ReshapeConfig {
                    shape: ReshapeInput::Static(shape),
                });
            }
            return Err(ProcessError::Custom(
                "Reshape: shape must be provided as either attribute or input".to_string(),
            ));
        }

        // Extract shape input as either static or runtime (opset 5+)
        let shape = match &node.inputs[1].ty {
            ArgType::Tensor(_tensor) => {
                // Extract shape from tensor input
                // Note: We don't validate rank here because extract_config runs before type inference
                // The rank might be 0 initially and will be updated during type inference
                match node.inputs[1].value() {
                    Some(tensor_data) => {
                        // Only validate when we have actual tensor data
                        assert_eq!(
                            tensor_data.shape.len(),
                            1,
                            "Reshape: shape tensor must be 1D"
                        );
                        ReshapeInput::Static(tensor_data.to_vec::<i64>().unwrap())
                    }
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
                    }
                }
            }
            ArgType::Shape(_) => {
                // Runtime input - store reference instead of cloning the argument
                ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        let config = ReshapeConfig { shape };
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Reshape(ReshapeNode {
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
    use crate::ir::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(allowzero: i64, shape_vec: Vec<i64>) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64_data("shape", shape_vec.clone(), vec![shape_vec.len()])
            .output_tensor_f32("reshaped", 2, None);

        if allowzero != 0 {
            builder = builder.attr_int("allowzero", allowzero);
        }

        builder
    }

    fn create_runtime_reshape_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Reshape, "test_runtime_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 0, None) // No static value - runtime input
            .output_tensor_f32("reshaped", 2, None)
    }

    fn create_reshape_with_shape_input() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Reshape, "test_reshape_with_shape")
            .input_tensor_f32("data", 4, None)
            .add_input("shape", ArgType::Shape(2))
            .output_tensor_f32("reshaped", 2, None)
    }

    #[test]
    fn test_reshape_config_basic() {
        let node = create_test_node(0, vec![2, 3]).process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.shape {
            ReshapeInput::Static(shape) => assert_eq!(shape, &vec![2, 3]),
            _ => panic!("Expected static shape"),
        }
    }

    #[test]
    fn test_reshape_config_allowzero_supported() {
        let _node = create_test_node(1, vec![2, 3]).process(ReshapeProcessor, 16);
        // Test passes if no panic occurs during processing
    }

    #[test]
    #[ignore] // TODO: Test needs redesign - runtime reshape requires rank information from output or Shape type input
    fn test_reshape_config_runtime() {
        let node = create_runtime_reshape_node().process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.shape {
            ReshapeInput::Runtime(runtime_ref) => assert_eq!(runtime_ref.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    fn test_reshape_config_no_shape_input() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);
        node.inputs.pop(); // Remove the shape input
        let processor = ReshapeProcessor;
        // With only 1 input and no shape attribute, extract_config should fail
        let result = processor.extract_config(&node, 16);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    #[should_panic(expected = "shape tensor must be 1D")]
    fn test_reshape_config_invalid_shape_dim() {
        // Create a node with 2D shape tensor (should trigger panic)
        let node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_with_data(
                "shape",
                DType::I64,
                2,                                                     // 2D tensor (rank 2)
                crate::ir::TensorData::new(vec![2i64, 3], vec![2, 1]), // 2D shape - this should cause panic
            )
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);
        let processor = ReshapeProcessor;
        // This should panic when validating the shape tensor is 1D
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_reshape_config_with_shape_type() {
        let node = create_reshape_with_shape_input().process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.shape {
            ReshapeInput::Runtime(runtime_ref) => assert_eq!(runtime_ref.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    fn test_reshape_dynamic_shape_with_output_rank() {
        // Test dynamic reshape where shape input has no static_shape,
        // but output rank is known from ONNX model metadata.
        // This simulates real-world models where shape is computed by other nodes (e.g., Concat)
        // but the ONNX model's value_info already specifies the output rank.

        let node = TestNodeBuilder::new(NodeType::Reshape, "test_dynamic_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 1, None) // Dynamic shape input - no static value
            .output_tensor_f32("reshaped", 4, None) // Output has rank 4 but no static_shape
            .build();

        // Verify the shape input has no static_shape (simulating dynamic case)
        match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.static_shape, None); // No static shape
            }
            _ => panic!("Expected tensor shape input"),
        }

        // Verify output has rank but no static_shape
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, None); // No static shape, just rank
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_scalar_to_neg1_keeps_scalar() {
        // Test that Reshape(scalar, [-1]) keeps output as Scalar
        // This optimization avoids unnecessary scalar -> tensor conversion
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape_scalar")
            .add_input("data", ArgType::ScalarNative(DType::F32))
            .input_tensor_i64_data("shape", vec![-1], vec![1])
            .add_output(
                "reshaped",
                ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
            )
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should remain scalar, not become a rank-1 tensor
        match &node.outputs[0].ty {
            ArgType::ScalarNative(dtype) => {
                assert_eq!(*dtype, DType::F32);
            }
            other => panic!("Expected Scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_reshape_scalar_to_1_keeps_scalar() {
        // Test that Reshape(scalar, [1]) keeps output as Scalar
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape_scalar_1")
            .add_input("data", ArgType::ScalarNative(DType::I64))
            .input_tensor_i64_data("shape", vec![1], vec![1])
            .add_output(
                "reshaped",
                ArgType::Tensor(TensorType::new(DType::I64, 1, None)),
            )
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should remain scalar
        match &node.outputs[0].ty {
            ArgType::ScalarNative(dtype) => {
                assert_eq!(*dtype, DType::I64);
            }
            other => panic!("Expected Scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_reshape_scalar_to_multi_element_becomes_tensor() {
        // Test that Reshape(scalar, [2]) does NOT keep scalar (would be invalid)
        // This ensures the optimization only applies to single-element shapes
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape_scalar_2")
            .add_input("data", ArgType::ScalarNative(DType::F32))
            .input_tensor_i64_data("shape", vec![2], vec![1])
            .add_output(
                "reshaped",
                ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
            )
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should be a tensor, not scalar (shape [2] means 2 elements)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 1);
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_reshape_is_noop_scalar_to_scalar() {
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape_noop")
            .add_input("data", ArgType::ScalarNative(DType::F32))
            .input_tensor_i64_data("shape", vec![-1], vec![1])
            .add_output(
                "reshaped",
                ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
            )
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Scalar reshaped to [-1] stays Scalar, so this is a no-op
        assert!(processor.is_noop(&node));
    }

    #[test]
    fn test_reshape_is_not_noop_tensor() {
        let node = create_test_node(0, vec![2, 3]).process(ReshapeProcessor, 16);
        let processor = ReshapeProcessor;

        // Tensor reshape is not a no-op
        assert!(!processor.is_noop(&node));
    }

    #[test]
    fn test_reshape_same_static_shape_is_noop() {
        let shape: Vec<Option<usize>> = vec![Some(2), Some(3), Some(4)];
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
            .output_tensor_f32("reshaped", 3, None)
            .build();

        // Set matching static shapes on input and output
        if let ArgType::Tensor(ref mut t) = node.inputs[0].ty {
            t.static_shape = Some(shape.clone());
        }
        if let ArgType::Tensor(ref mut t) = node.outputs[0].ty {
            t.static_shape = Some(shape);
        }

        let processor = ReshapeProcessor;
        assert!(processor.is_noop(&node));
    }

    #[test]
    fn test_reshape_different_static_shape_is_not_noop() {
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 3, None)
            .input_tensor_i64_data("shape", vec![6, 4], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build();

        if let ArgType::Tensor(ref mut t) = node.inputs[0].ty {
            t.static_shape = Some(vec![Some(2), Some(3), Some(4)]);
        }
        if let ArgType::Tensor(ref mut t) = node.outputs[0].ty {
            t.static_shape = Some(vec![Some(6), Some(4)]);
        }

        let processor = ReshapeProcessor;
        assert!(!processor.is_noop(&node));
    }

    #[test]
    fn test_reshape_static_shape_positive_values() {
        // Input [2, 3, 4], reshape to [6, 4]
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 3, Some(vec![2, 3, 4]))
            .input_tensor_i64_data("shape", vec![6, 4], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(6), Some(4)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_static_shape_with_neg1() {
        // Input [2, 3, 4] (total=24), reshape to [4, -1] -> [4, 6]
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 3, Some(vec![2, 3, 4]))
            .input_tensor_i64_data("shape", vec![4, -1], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(4), Some(6)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_static_shape_with_zero() {
        // Input [2, 3, 4], reshape to [0, 12] -> [2, 12] (0 copies from input)
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 3, Some(vec![2, 3, 4]))
            .input_tensor_i64_data("shape", vec![0, 12], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(12)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_static_shape_unknown_input() {
        // Input with no static_shape, reshape to [3, 4]
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("shape", vec![3, 4], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                // Positive values are always known, 0 needs input, -1 needs input total
                assert_eq!(t.static_shape, Some(vec![Some(3), Some(4)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_neg1_without_input_shape() {
        // Input with no static_shape, reshape to [-1, 4] -> -1 can't be resolved
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("shape", vec![-1, 4], vec![2])
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![None, Some(4)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
