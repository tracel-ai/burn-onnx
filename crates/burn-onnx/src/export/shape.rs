use burn::backend::ir::{
    ActivationOperationIr, BaseOperationIr, FloatOperationIr, GraphIr, IntOperationIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, TensorId, TensorIr,
};
use hashbrown::HashSet;

use crate::export::{
    DynamicAxis, ExportError, GraphStructureValidator, ResolvedExportGraph, ResolvedShape,
    ShapeExpr,
};

use super::lower::patterns;

/// Annotation for one runtime input axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSpec {
    /// The dimension is fixed to the captured value.
    Static,
    /// The dimension is runtime-variable and identified by an ONNX symbol.
    Dynamic {
        /// ONNX symbolic dimension name.
        symbol: String,
    },
}

/// Shape annotations for one graph input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    /// Axis annotations in tensor order.
    pub axes: Vec<AxisSpec>,
}

impl InputSpec {
    /// Create annotations for one runtime input, in input-axis order.
    pub fn new(axes: impl Into<Vec<AxisSpec>>) -> Self {
        Self { axes: axes.into() }
    }
}

impl AxisSpec {
    /// Create a dynamic axis annotation.
    pub fn dynamic(symbol: impl Into<String>) -> Self {
        Self::Dynamic {
            symbol: symbol.into(),
        }
    }
}

/// Replaceable shape-resolution stage used before ONNX lowering.
pub(crate) trait ShapeResolver {
    /// Resolve shape-sensitive operands in a captured graph.
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError>;
}

/// Resolves every captured shape-sensitive operation as constants.
pub(crate) struct StaticShapeResolver<'a> {
    /// Captured graph.
    pub(crate) graph: &'a GraphIr,
}

impl ShapeResolver for StaticShapeResolver<'_> {
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError> {
        Ok(ResolvedExportGraph {
            graph: self.graph.clone(),
            shapes: shape_operations(self.graph)
                .map(|operation| ResolvedShape {
                    operation: operation.index,
                    tensor: operation.output.id,
                    dimensions: operation
                        .output
                        .shape
                        .iter()
                        .copied()
                        .map(ShapeExpr::Static)
                        .collect(),
                })
                .collect(),
            dynamic_axes: Vec::new(),
        })
    }
}

/// Conservative shape resolver using two structurally validated captures.
pub(crate) struct PairedTraceShapeResolver<'a> {
    /// Primary capture.
    pub(crate) sample: &'a GraphIr,
    /// Capture produced with validation input dimensions.
    pub(crate) validation: &'a GraphIr,
    /// Explicit dynamic input annotations.
    pub(crate) inputs: &'a [InputSpec],
}

impl ShapeResolver for PairedTraceShapeResolver<'_> {
    fn resolve(&self) -> Result<ResolvedExportGraph, ExportError> {
        GraphStructureValidator::validate(self.sample, self.validation)?;
        validate_shape_sensitive_operations(self.sample, self.validation)?;
        let sample_shapes = boundary_shapes(self.sample, &self.sample.inputs)?;
        let validation_shapes = boundary_shapes(self.validation, &self.validation.inputs)?;
        validate_input_specs(self.inputs, &sample_shapes, &validation_shapes)?;
        let mut dynamic_axes = Vec::new();
        for (position, spec) in self.inputs.iter().enumerate() {
            let sample_id = self.sample.inputs[position];
            let validation_id = self.validation.inputs[position];
            let sample_input =
                tensor(self.sample, sample_id).ok_or(ExportError::MissingValue(sample_id))?;
            let validation_input = tensor(self.validation, validation_id)
                .ok_or(ExportError::MissingValue(validation_id))?;
            if spec.axes.len() != sample_input.shape.len() {
                return Err(ExportError::InvalidBoundary(format!(
                    "input {position} has rank {} but its spec has {} axes",
                    sample_input.shape.len(),
                    spec.axes.len()
                )));
            }
            for (axis, axis_spec) in spec.axes.iter().enumerate() {
                match axis_spec {
                    AxisSpec::Static => {
                        if sample_input.shape[axis] != validation_input.shape[axis] {
                            return Err(ExportError::DynamicGraphMismatch {
                                operation: 0,
                                reason: format!(
                                    "static input {position} axis {axis} differs ({} != {})",
                                    sample_input.shape[axis], validation_input.shape[axis]
                                ),
                            });
                        }
                    }
                    AxisSpec::Dynamic { symbol } => {
                        dynamic_axes.push(DynamicAxis {
                            tensor: sample_id,
                            axis,
                            symbol: symbol.clone(),
                        });
                    }
                }
            }
        }
        let potentially_dynamic =
            potentially_dynamic_axes(self.sample, self.validation, self.inputs);
        for (position, (&sample_id, &validation_id)) in self
            .sample
            .outputs
            .iter()
            .zip(&self.validation.outputs)
            .enumerate()
        {
            let sample_output =
                tensor(self.sample, sample_id).ok_or(ExportError::MissingValue(sample_id))?;
            let validation_output = tensor(self.validation, validation_id)
                .ok_or(ExportError::MissingValue(validation_id))?;
            for (axis, (&sample_dim, &validation_dim)) in sample_output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                if !potentially_dynamic.contains(&(sample_id, axis)) {
                    continue;
                }
                let mut symbols = Vec::new();
                for (input_position, spec) in self.inputs.iter().enumerate() {
                    let sample_input = tensor(self.sample, self.sample.inputs[input_position])
                        .ok_or(ExportError::MissingValue(
                            self.sample.inputs[input_position],
                        ))?;
                    let validation_input =
                        tensor(self.validation, self.validation.inputs[input_position]).ok_or(
                            ExportError::MissingValue(self.validation.inputs[input_position]),
                        )?;
                    for (input_axis, axis_spec) in spec.axes.iter().enumerate() {
                        if let AxisSpec::Dynamic { symbol, .. } = axis_spec
                            && sample_input.shape[input_axis] == sample_dim
                            && validation_input.shape[input_axis] == validation_dim
                        {
                            symbols.push(symbol.clone());
                        }
                    }
                }
                symbols.sort();
                symbols.dedup();
                if dynamic_axes
                    .iter()
                    .any(|dynamic| dynamic.tensor == sample_id && dynamic.axis == axis)
                {
                    continue;
                }
                dynamic_axes.push(DynamicAxis {
                    tensor: sample_id,
                    axis,
                    symbol: if symbols.len() == 1 {
                        symbols.pop().unwrap()
                    } else {
                        format!("output_{position}_dim_{axis}")
                    },
                });
            }
        }
        let validation_shape_operations: Vec<_> = shape_operations(self.validation).collect();
        let mut shapes = Vec::new();
        for (sample_operation, validation_operation) in
            shape_operations(self.sample).zip(validation_shape_operations)
        {
            let operation = sample_operation.index;
            let output = sample_operation.output;
            let validation_output = validation_operation.output;
            let mut dimensions = Vec::new();
            let mut unresolved = Vec::new();
            for (axis, (&sample_dim, &validation_dim)) in output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                let mut candidates = Vec::new();
                for (position, spec) in self.inputs.iter().enumerate() {
                    let sample_id = self.sample.inputs[position];
                    if let (Some(sample_input), Some(validation_input)) = (
                        tensor(self.sample, sample_id),
                        tensor(self.validation, self.validation.inputs[position]),
                    ) {
                        for (input_axis, axis_spec) in spec.axes.iter().enumerate() {
                            if matches!(axis_spec, AxisSpec::Dynamic { .. })
                                && sample_input.shape.get(input_axis) == Some(&sample_dim)
                                && validation_input.shape.get(input_axis) == Some(&validation_dim)
                            {
                                candidates.push(ShapeExpr::InputDim {
                                    input: sample_id,
                                    axis: input_axis,
                                });
                            }
                        }
                    }
                }
                if candidates.is_empty()
                    && let (Some(source), Some(validation_source)) =
                        (sample_operation.source, validation_operation.source)
                {
                    for (source_axis, (&source_dim, &validation_source_dim)) in source
                        .shape
                        .iter()
                        .zip(validation_source.shape.iter())
                        .enumerate()
                    {
                        if potentially_dynamic.contains(&(source.id, source_axis))
                            && source_dim == sample_dim
                            && validation_source_dim == validation_dim
                        {
                            candidates.push(ShapeExpr::TensorDim {
                                tensor: source.id,
                                axis: source_axis,
                            });
                        }
                    }
                }
                candidates.dedup();
                match candidates.len() {
                    1 => dimensions.push(candidates.pop().unwrap()),
                    0 if sample_dim == validation_dim => {
                        if let Some(source) = sample_operation.source {
                            let has_static_source_match = source.shape.iter().enumerate().any(
                                |(source_axis, &source_dim)| {
                                    source_dim == sample_dim
                                        && !potentially_dynamic.contains(&(source.id, source_axis))
                                },
                            );
                            let has_dynamic_source =
                                source.shape.iter().enumerate().any(|(source_axis, _)| {
                                    potentially_dynamic.contains(&(source.id, source_axis))
                                });
                            if has_static_source_match || !has_dynamic_source {
                                dimensions.push(ShapeExpr::Static(sample_dim));
                            } else {
                                unresolved.push(axis);
                                dimensions.push(ShapeExpr::Infer);
                            }
                        } else if aligned_static_input_dimension(
                            sample_dim,
                            axis,
                            output.shape.num_dims(),
                            self.sample,
                            self.inputs,
                        ) || !has_dynamic_inputs(self.inputs)
                        {
                            dimensions.push(ShapeExpr::Static(sample_dim));
                        } else {
                            return Err(ExportError::DynamicShapeLost {
                                tensor: output.id,
                                axis,
                                reason: "equal Full dimensions cannot be proven static from two captures"
                                    .into(),
                            });
                        }
                    }
                    0 if sample_operation.source.is_none() => {
                        return Err(ExportError::DynamicShapeLost {
                            tensor: output.id,
                            axis,
                            reason: "Full dimension does not match an annotated dynamic input axis"
                                .into(),
                        });
                    }
                    0 => {
                        unresolved.push(axis);
                        dimensions.push(ShapeExpr::Infer);
                    }
                    count => {
                        return Err(ExportError::DynamicShapeLost {
                            tensor: output.id,
                            axis,
                            reason: format!(
                                "dimension matches {count} dynamic source axes and is ambiguous"
                            ),
                        });
                    }
                }
            }
            if unresolved.len() > 1 {
                let axis = unresolved[0];
                return Err(ExportError::DynamicShapeLost {
                    tensor: output.id,
                    axis,
                    reason: "multiple element-count-derived dimensions remain".into(),
                });
            }
            shapes.push(ResolvedShape {
                operation,
                tensor: output.id,
                dimensions,
            });
        }
        Ok(ResolvedExportGraph {
            graph: self.sample.clone(),
            shapes,
            dynamic_axes,
        })
    }
}

/// Reject varying shape operands that the paired resolver cannot represent.
///
/// Structural validation deliberately ignores these values. Every ignored
/// operand must be handled here so lowering can never silently embed a value
/// taken only from the sample capture.
fn validate_shape_sensitive_operations(
    sample: &GraphIr,
    validation: &GraphIr,
) -> Result<(), ExportError> {
    for (index, (sample_operation, validation_operation)) in sample
        .operations
        .iter()
        .zip(&validation.operations)
        .enumerate()
    {
        match (sample_operation, validation_operation) {
            (
                OperationIr::Module(ModuleOperationIr::Interpolate(sample)),
                OperationIr::Module(ModuleOperationIr::Interpolate(validation)),
            ) if sample.output_size != validation.output_size => {
                let offset = sample
                    .output_size
                    .iter()
                    .zip(validation.output_size)
                    .position(|(sample, validation)| *sample != validation)
                    .unwrap_or(0);
                let axis = sample.out.shape.num_dims().saturating_sub(2) + offset;
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!("operation {index} has a varying interpolation output size"),
                });
            }
            (
                OperationIr::BaseFloat(BaseOperationIr::Slice(sample))
                | OperationIr::BaseInt(BaseOperationIr::Slice(sample))
                | OperationIr::BaseBool(BaseOperationIr::Slice(sample)),
                OperationIr::BaseFloat(BaseOperationIr::Slice(validation))
                | OperationIr::BaseInt(BaseOperationIr::Slice(validation))
                | OperationIr::BaseBool(BaseOperationIr::Slice(validation)),
            ) if sample.ranges != validation.ranges => {
                let axis = sample
                    .ranges
                    .iter()
                    .zip(&validation.ranges)
                    .position(|(sample, validation)| sample != validation)
                    .unwrap_or(0);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample.out.id,
                    axis,
                    reason: format!("operation {index} has varying slice bounds"),
                });
            }
            (
                OperationIr::BaseFloat(BaseOperationIr::SliceAssign(sample_slice))
                | OperationIr::BaseInt(BaseOperationIr::SliceAssign(sample_slice)),
                OperationIr::BaseFloat(BaseOperationIr::SliceAssign(validation_slice))
                | OperationIr::BaseInt(BaseOperationIr::SliceAssign(validation_slice)),
            ) if sample_slice.ranges != validation_slice.ranges => {
                if matches!(
                    (
                        patterns::constant_pad(&sample.operations, index),
                        patterns::constant_pad(&validation.operations, index),
                    ),
                    (Some(sample_pad), Some(validation_pad))
                        if sample_pad.pads == validation_pad.pads
                ) {
                    continue;
                }
                let axis = sample_slice
                    .ranges
                    .iter()
                    .zip(&validation_slice.ranges)
                    .position(|(sample, validation)| sample != validation)
                    .unwrap_or(0);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample_slice.out.id,
                    axis,
                    reason: format!("operation {index} has varying slice-assignment bounds"),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Conservatively propagate axes whose sizes can depend on a declared dynamic input.
///
/// Differing observations prove an axis is dynamic. Operation-specific transfer rules retain
/// dependencies when two observations happen to produce the same concrete size. Reshape is
/// handled separately because it can move axes or combine them.
fn potentially_dynamic_axes(
    sample: &GraphIr,
    validation: &GraphIr,
    specs: &[InputSpec],
) -> HashSet<(TensorId, usize)> {
    let mut dynamic = HashSet::new();
    for (position, spec) in specs.iter().enumerate() {
        let Some(&input) = sample.inputs.get(position) else {
            continue;
        };
        for (axis, axis_spec) in spec.axes.iter().enumerate() {
            if matches!(axis_spec, AxisSpec::Dynamic { .. }) {
                dynamic.insert((input, axis));
            }
        }
    }

    for (sample_operation, validation_operation) in
        sample.operations.iter().zip(&validation.operations)
    {
        let sample_outputs = sample_operation.outputs().collect::<Vec<_>>();
        let validation_outputs = validation_operation.outputs().collect::<Vec<_>>();
        for (sample_output, validation_output) in sample_outputs.iter().zip(&validation_outputs) {
            for (axis, (&sample_dim, &validation_dim)) in sample_output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                if sample_dim != validation_dim {
                    dynamic.insert((sample_output.id, axis));
                }
            }
        }

        propagate_dynamic_axes(sample_operation, &mut dynamic);

        let (Some((source, output)), Some((validation_source, validation_output))) = (
            reshape_tensors(sample_operation),
            reshape_tensors(validation_operation),
        ) else {
            continue;
        };
        let source_has_dynamic = source
            .shape
            .iter()
            .enumerate()
            .any(|(axis, _)| dynamic.contains(&(source.id, axis)));
        for (output_axis, (&output_dim, &validation_output_dim)) in output
            .shape
            .iter()
            .zip(validation_output.shape.iter())
            .enumerate()
        {
            if dynamic.contains(&(output.id, output_axis)) {
                continue;
            }
            let dynamic_source_match = source
                .shape
                .iter()
                .zip(validation_source.shape.iter())
                .enumerate()
                .any(|(source_axis, (&source_dim, &validation_source_dim))| {
                    dynamic.contains(&(source.id, source_axis))
                        && source_dim == output_dim
                        && validation_source_dim == validation_output_dim
                });
            let static_source_match =
                source
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(source_axis, &source_dim)| {
                        !dynamic.contains(&(source.id, source_axis)) && source_dim == output_dim
                    });
            if dynamic_source_match || (source_has_dynamic && !static_source_match) {
                dynamic.insert((output.id, output_axis));
            }
        }
    }
    dynamic
}

/// Transfer dynamic-axis dependencies according to the shape semantics of exported operations.
///
/// Operations not listed here only gain dynamic axes from differing trace observations. This is
/// deliberate: assuming that every equal-rank operation preserves axis positions incorrectly
/// marks fixed dimensions for operations such as linear layers and adaptive pooling.
fn propagate_dynamic_axes(operation: &OperationIr, dynamic: &mut HashSet<(TensorId, usize)>) {
    match operation {
        OperationIr::BaseFloat(operation)
        | OperationIr::BaseInt(operation)
        | OperationIr::BaseBool(operation) => match operation {
            BaseOperationIr::SwapDims(operation) => {
                for output_axis in 0..operation.out.shape.num_dims() {
                    let input_axis = if output_axis == operation.dim1 {
                        operation.dim2
                    } else if output_axis == operation.dim2 {
                        operation.dim1
                    } else {
                        output_axis
                    };
                    propagate_axis(
                        &operation.input,
                        input_axis,
                        &operation.out,
                        output_axis,
                        dynamic,
                    );
                }
            }
            BaseOperationIr::Permute(operation) => {
                for (output_axis, &input_axis) in operation.axes.iter().enumerate() {
                    propagate_axis(
                        &operation.input,
                        input_axis,
                        &operation.out,
                        output_axis,
                        dynamic,
                    );
                }
            }
            BaseOperationIr::Flip(operation) => {
                propagate_same_axes(&operation.input, &operation.out, dynamic);
            }
            BaseOperationIr::Expand(operation) => {
                propagate_broadcast_axes(&operation.input, &operation.out, dynamic);
            }
            BaseOperationIr::RepeatDim(operation) => {
                propagate_same_axes(&operation.tensor, &operation.out, dynamic);
            }
            BaseOperationIr::Cat(operation) => {
                for input in &operation.tensors {
                    propagate_same_axes(input, &operation.out, dynamic);
                }
            }
            BaseOperationIr::Cast(operation) => {
                propagate_same_axes(&operation.input, &operation.out, dynamic);
            }
            BaseOperationIr::AllDim(operation) | BaseOperationIr::AnyDim(operation) => {
                propagate_reduced_axes(&operation.input, &operation.out, operation.axis, dynamic);
            }
            // Reshape has a separate value-aware transfer below. Other base operations are not
            // currently lowered directly by this exporter.
            _ => {}
        },
        OperationIr::NumericFloat(_, operation) | OperationIr::NumericInt(_, operation) => {
            match operation {
                NumericOperationIr::Add(operation)
                | NumericOperationIr::Sub(operation)
                | NumericOperationIr::Mul(operation)
                | NumericOperationIr::Div(operation) => {
                    propagate_broadcast_axes(&operation.lhs, &operation.out, dynamic);
                    propagate_broadcast_axes(&operation.rhs, &operation.out, dynamic);
                }
                NumericOperationIr::AddScalar(operation)
                | NumericOperationIr::SubScalar(operation)
                | NumericOperationIr::MulScalar(operation)
                | NumericOperationIr::DivScalar(operation) => {
                    propagate_same_axes(&operation.lhs, &operation.out, dynamic);
                }
                NumericOperationIr::Abs(operation) | NumericOperationIr::Neg(operation) => {
                    propagate_same_axes(&operation.input, &operation.out, dynamic);
                }
                NumericOperationIr::MeanDim(operation)
                | NumericOperationIr::SumDim(operation)
                | NumericOperationIr::ProdDim(operation) => {
                    propagate_reduced_axes(
                        &operation.input,
                        &operation.out,
                        operation.axis,
                        dynamic,
                    );
                }
                _ => {}
            }
        }
        OperationIr::Float(_, operation) => match operation {
            FloatOperationIr::Exp(operation)
            | FloatOperationIr::Log(operation)
            | FloatOperationIr::Sqrt(operation)
            | FloatOperationIr::Tanh(operation) => {
                propagate_same_axes(&operation.input, &operation.out, dynamic);
            }
            FloatOperationIr::Matmul(operation) => {
                propagate_matmul_axes(&operation.lhs, &operation.rhs, &operation.out, dynamic);
            }
            _ => {}
        },
        OperationIr::Int(IntOperationIr::Matmul(operation)) => {
            propagate_matmul_axes(&operation.lhs, &operation.rhs, &operation.out, dynamic);
        }
        OperationIr::Activation(
            ActivationOperationIr::Relu(operation) | ActivationOperationIr::Sigmoid(operation),
        ) => {
            propagate_same_axes(&operation.input, &operation.out, dynamic);
        }
        OperationIr::Module(operation) => match operation {
            ModuleOperationIr::Conv2d(operation) => {
                propagate_axis(&operation.x, 0, &operation.out, 0, dynamic);
                propagate_axis(&operation.x, 2, &operation.out, 2, dynamic);
                propagate_axis(&operation.x, 3, &operation.out, 3, dynamic);
            }
            ModuleOperationIr::BatchNorm(operation) => {
                propagate_same_axes(&operation.x, &operation.out, dynamic);
            }
            ModuleOperationIr::Interpolate(operation) => {
                propagate_axis(&operation.x, 0, &operation.out, 0, dynamic);
                propagate_axis(&operation.x, 1, &operation.out, 1, dynamic);
            }
            ModuleOperationIr::AdaptiveAvgPool2d(operation) => {
                propagate_axis(&operation.x, 0, &operation.out, 0, dynamic);
                propagate_axis(&operation.x, 1, &operation.out, 1, dynamic);
            }
            ModuleOperationIr::MaxPool2d(operation) => {
                propagate_same_axes(&operation.x, &operation.out, dynamic);
            }
            ModuleOperationIr::AvgPool2d(operation) => {
                propagate_same_axes(&operation.x, &operation.out, dynamic);
            }
            ModuleOperationIr::Linear(operation) => {
                for axis in 0..operation.out.shape.num_dims().saturating_sub(1) {
                    propagate_axis(&operation.x, axis, &operation.out, axis, dynamic);
                }
            }
            _ => {}
        },
        _ => {}
    }
}

fn propagate_same_axes(
    input: &TensorIr,
    output: &TensorIr,
    dynamic: &mut HashSet<(TensorId, usize)>,
) {
    for axis in 0..input.shape.num_dims().min(output.shape.num_dims()) {
        propagate_axis(input, axis, output, axis, dynamic);
    }
}

fn propagate_broadcast_axes(
    input: &TensorIr,
    output: &TensorIr,
    dynamic: &mut HashSet<(TensorId, usize)>,
) {
    let Some(offset) = output.shape.num_dims().checked_sub(input.shape.num_dims()) else {
        return;
    };
    for input_axis in 0..input.shape.num_dims() {
        propagate_axis(input, input_axis, output, offset + input_axis, dynamic);
    }
}

fn propagate_reduced_axes(
    input: &TensorIr,
    output: &TensorIr,
    reduced_axis: usize,
    dynamic: &mut HashSet<(TensorId, usize)>,
) {
    for axis in 0..input.shape.num_dims().min(output.shape.num_dims()) {
        if axis != reduced_axis {
            propagate_axis(input, axis, output, axis, dynamic);
        }
    }
}

fn propagate_matmul_axes(
    lhs: &TensorIr,
    rhs: &TensorIr,
    output: &TensorIr,
    dynamic: &mut HashSet<(TensorId, usize)>,
) {
    let output_rank = output.shape.num_dims();
    let lhs_rank = lhs.shape.num_dims();
    let rhs_rank = rhs.shape.num_dims();
    if output_rank < 2 || lhs_rank < 2 || rhs_rank < 2 {
        return;
    }

    let output_batch_rank = output_rank - 2;
    let lhs_batch_rank = lhs_rank - 2;
    let rhs_batch_rank = rhs_rank - 2;
    for axis in 0..lhs_batch_rank {
        propagate_axis(
            lhs,
            axis,
            output,
            output_batch_rank - lhs_batch_rank + axis,
            dynamic,
        );
    }
    for axis in 0..rhs_batch_rank {
        propagate_axis(
            rhs,
            axis,
            output,
            output_batch_rank - rhs_batch_rank + axis,
            dynamic,
        );
    }
    propagate_axis(lhs, lhs_rank - 2, output, output_rank - 2, dynamic);
    propagate_axis(rhs, rhs_rank - 1, output, output_rank - 1, dynamic);
}

fn propagate_axis(
    input: &TensorIr,
    input_axis: usize,
    output: &TensorIr,
    output_axis: usize,
    dynamic: &mut HashSet<(TensorId, usize)>,
) {
    if dynamic.contains(&(input.id, input_axis)) {
        dynamic.insert((output.id, output_axis));
    }
}

fn reshape_tensors(
    operation: &OperationIr,
) -> Option<(&burn::backend::ir::TensorIr, &burn::backend::ir::TensorIr)> {
    match operation {
        OperationIr::BaseFloat(BaseOperationIr::Reshape(operation))
        | OperationIr::BaseInt(BaseOperationIr::Reshape(operation))
        | OperationIr::BaseBool(BaseOperationIr::Reshape(operation)) => {
            Some((&operation.input, &operation.out))
        }
        _ => None,
    }
}

fn aligned_static_input_dimension(
    value: usize,
    axis: usize,
    rank: usize,
    sample: &GraphIr,
    specs: &[InputSpec],
) -> bool {
    specs.iter().enumerate().any(|(position, spec)| {
        let Some(input) = sample
            .inputs
            .get(position)
            .and_then(|&input| tensor(sample, input))
        else {
            return false;
        };
        input.shape.num_dims() == rank
            && matches!(spec.axes.get(axis), Some(AxisSpec::Static))
            && input.shape.get(axis) == Some(&value)
    })
}

fn has_dynamic_inputs(specs: &[InputSpec]) -> bool {
    specs
        .iter()
        .flat_map(|spec| &spec.axes)
        .any(|axis| matches!(axis, AxisSpec::Dynamic { .. }))
}

pub(crate) fn validate_input_specs(
    specs: &[InputSpec],
    sample_shapes: &[Vec<usize>],
    validation_shapes: &[Vec<usize>],
) -> Result<(), ExportError> {
    if sample_shapes.len() != validation_shapes.len() {
        return Err(ExportError::InvalidBoundary(format!(
            "sample inputs contain {} tensors but validation inputs contain {}",
            sample_shapes.len(),
            validation_shapes.len()
        )));
    }
    if specs.len() != sample_shapes.len() {
        return Err(ExportError::InvalidBoundary(format!(
            "received {} input specs for {} input tensors",
            specs.len(),
            sample_shapes.len()
        )));
    }

    let mut symbols = hashbrown::HashMap::<&str, (usize, usize)>::new();
    for (input, ((spec, sample), validation)) in specs
        .iter()
        .zip(sample_shapes)
        .zip(validation_shapes)
        .enumerate()
    {
        if sample.len() != validation.len() {
            return Err(ExportError::InvalidBoundary(format!(
                "sample input {input} has rank {} but validation input has rank {}",
                sample.len(),
                validation.len()
            )));
        }
        if spec.axes.len() != sample.len() {
            return Err(ExportError::InvalidBoundary(format!(
                "input {input} has rank {} but its spec has {} axes",
                sample.len(),
                spec.axes.len()
            )));
        }
        for (axis, ((axis_spec, &sample_dim), &validation_dim)) in
            spec.axes.iter().zip(sample).zip(validation).enumerate()
        {
            match axis_spec {
                AxisSpec::Static if sample_dim != validation_dim => {
                    return Err(ExportError::InvalidBoundary(format!(
                        "static input {input} axis {axis} differs ({sample_dim} != {validation_dim})"
                    )));
                }
                AxisSpec::Dynamic { symbol } => {
                    if symbol.is_empty() {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic input {input} axis {axis} has an empty symbol"
                        )));
                    }
                    if sample_dim == validation_dim {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic input {input} axis {axis} must differ between sample and validation inputs"
                        )));
                    }
                    if let Some((previous_sample, previous_validation)) =
                        symbols.insert(symbol, (sample_dim, validation_dim))
                        && (previous_sample, previous_validation) != (sample_dim, validation_dim)
                    {
                        return Err(ExportError::InvalidBoundary(format!(
                            "dynamic symbol `{symbol}` refers to inconsistent dimensions"
                        )));
                    }
                }
                AxisSpec::Static => {}
            }
        }
    }
    Ok(())
}

fn boundary_shapes(graph: &GraphIr, ids: &[TensorId]) -> Result<Vec<Vec<usize>>, ExportError> {
    ids.iter()
        .map(|&id| {
            tensor(graph, id)
                .map(|tensor| tensor.shape.to_vec())
                .ok_or(ExportError::MissingValue(id))
        })
        .collect()
}

struct ShapeOperation<'a> {
    index: usize,
    source: Option<&'a burn::backend::ir::TensorIr>,
    output: &'a burn::backend::ir::TensorIr,
}

fn shape_operations(graph: &GraphIr) -> impl Iterator<Item = ShapeOperation<'_>> {
    graph
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| match operation {
            OperationIr::BaseFloat(BaseOperationIr::Reshape(op))
            | OperationIr::BaseInt(BaseOperationIr::Reshape(op))
            | OperationIr::BaseBool(BaseOperationIr::Reshape(op)) => Some(ShapeOperation {
                index,
                source: Some(&op.input),
                output: &op.out,
            }),
            OperationIr::NumericFloat(_, NumericOperationIr::Full(op))
            | OperationIr::NumericInt(_, NumericOperationIr::Full(op)) => {
                (!patterns::is_constant_pad_full(&graph.operations, index)).then_some(
                    ShapeOperation {
                        index,
                        source: None,
                        output: &op.out,
                    },
                )
            }
            _ => None,
        })
}

fn tensor(graph: &GraphIr, id: TensorId) -> Option<&burn::backend::ir::TensorIr> {
    graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .find(|tensor| tensor.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ir::{
        AdaptiveAvgPool2dOpIr, FullOpIr, InterpolateModeIr, InterpolateOpIr, InterpolateOptionsIr,
        MatmulOpIr, ScalarIr, ShapeOpIr, SwapDimsOpIr, TensorIr,
    };
    use burn::backend::{DType, Shape};

    fn tensor(id: u64, shape: &[usize]) -> TensorIr {
        TensorIr::uninit(
            TensorId::new(id),
            shape.iter().copied().collect::<Shape>(),
            DType::F32,
        )
    }

    fn reshape(input_id: u64, output_id: u64, input: &[usize], output: &[usize]) -> GraphIr {
        GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::Reshape(
            ShapeOpIr {
                input: tensor(input_id, input),
                out: tensor(output_id, output),
            },
        ))])
    }

    #[test]
    fn static_resolver_emits_constants() {
        let graph = reshape(1, 2, &[2, 3, 4], &[2, 12]);
        let resolved = StaticShapeResolver { graph: &graph }.resolve().unwrap();
        assert_eq!(
            resolved.shapes[0].dimensions,
            vec![ShapeExpr::Static(2), ShapeExpr::Static(12)]
        );
    }

    #[test]
    fn paired_resolver_uses_input_dim_and_infer() {
        let sample = reshape(1, 2, &[2, 3, 4, 5], &[2, 3, 20]);
        // Deliberately use different tensor IDs to exercise structural normalization.
        let validation = reshape(11, 12, &[7, 3, 6, 7], &[7, 3, 42]);
        let specs = [InputSpec::new(vec![
            AxisSpec::Dynamic { symbol: "N".into() },
            AxisSpec::Static,
            AxisSpec::Dynamic { symbol: "H".into() },
            AxisSpec::Dynamic { symbol: "W".into() },
        ])];
        let resolved = PairedTraceShapeResolver {
            sample: &sample,
            validation: &validation,
            inputs: &specs,
        }
        .resolve()
        .unwrap();
        assert_eq!(
            resolved.shapes[0].dimensions,
            vec![
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 0
                },
                ShapeExpr::Static(3),
                ShapeExpr::Infer,
            ]
        );
    }

    #[test]
    fn paired_resolver_infers_unproven_inserted_axis() {
        let sample = reshape(1, 2, &[2, 5, 7], &[2, 1, 5, 7]);
        let validation = reshape(11, 12, &[3, 6, 8], &[3, 1, 6, 8]);
        let specs = [InputSpec::new([
            AxisSpec::dynamic("N"),
            AxisSpec::dynamic("H"),
            AxisSpec::dynamic("W"),
        ])];
        let resolved = PairedTraceShapeResolver {
            sample: &sample,
            validation: &validation,
            inputs: &specs,
        }
        .resolve()
        .unwrap();
        assert_eq!(
            resolved.shapes[0].dimensions,
            vec![
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 0,
                },
                ShapeExpr::Infer,
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 1,
                },
                ShapeExpr::InputDim {
                    input: TensorId::new(1),
                    axis: 2,
                },
            ]
        );
    }

    #[test]
    fn paired_resolver_rejects_coincident_dynamic_axes() {
        let sample = reshape(1, 2, &[2, 2], &[2, 2]);
        let validation = reshape(11, 12, &[3, 3], &[3, 3]);
        let specs = [InputSpec::new([
            AxisSpec::dynamic("first"),
            AxisSpec::dynamic("second"),
        ])];
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &specs,
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost { .. })
        ));
    }

    #[test]
    fn paired_resolver_rejects_unresolved_dynamic_full_shape() {
        let sample = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Full(FullOpIr {
                out: tensor(1, &[2, 3]),
                value: ScalarIr::Float(0.0),
            }),
        )]);
        let validation = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Full(FullOpIr {
                out: tensor(11, &[5, 3]),
                value: ScalarIr::Float(0.0),
            }),
        )]);

        GraphStructureValidator::validate(&sample, &validation).unwrap();
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &[],
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost {
                tensor,
                axis: 0,
                ..
            }) if tensor == TensorId::new(1)
        ));
    }

    #[test]
    fn paired_resolver_rejects_varying_interpolation_size() {
        let sample = GraphIr::new(vec![OperationIr::Module(ModuleOperationIr::Interpolate(
            InterpolateOpIr {
                x: tensor(1, &[1, 1, 4, 4]),
                output_size: [8, 8],
                options: InterpolateOptionsIr {
                    mode: InterpolateModeIr::Nearest,
                    align_corners: false,
                },
                out: tensor(2, &[1, 1, 8, 8]),
            },
        ))]);
        let validation = GraphIr::new(vec![OperationIr::Module(ModuleOperationIr::Interpolate(
            InterpolateOpIr {
                x: tensor(11, &[1, 1, 5, 5]),
                output_size: [10, 10],
                options: InterpolateOptionsIr {
                    mode: InterpolateModeIr::Nearest,
                    align_corners: false,
                },
                out: tensor(12, &[1, 1, 10, 10]),
            },
        ))]);

        GraphStructureValidator::validate(&sample, &validation).unwrap();
        assert!(matches!(
            (PairedTraceShapeResolver {
                sample: &sample,
                validation: &validation,
                inputs: &[],
            })
            .resolve(),
            Err(ExportError::DynamicShapeLost { tensor, axis: 2, .. })
                if tensor == TensorId::new(2)
        ));
    }

    #[test]
    fn dynamic_axes_follow_swapped_axis_positions_without_marking_the_old_position() {
        let sample_input = tensor(1, &[2, 3]);
        let sample_output = tensor(2, &[3, 2]);
        let validation_input = tensor(11, &[4, 3]);
        let validation_output = tensor(12, &[3, 4]);
        let mut sample = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: sample_input,
                out: sample_output,
                dim1: 0,
                dim2: 1,
            },
        ))]);
        sample.inputs = vec![TensorId::new(1)];
        sample.outputs = vec![TensorId::new(2)];
        let mut validation = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: validation_input,
                out: validation_output,
                dim1: 0,
                dim2: 1,
            },
        ))]);
        validation.inputs = vec![TensorId::new(11)];
        validation.outputs = vec![TensorId::new(12)];
        let specs = [InputSpec::new([
            AxisSpec::dynamic("rows"),
            AxisSpec::Static,
        ])];

        let dynamic = potentially_dynamic_axes(&sample, &validation, &specs);

        assert!(dynamic.contains(&(TensorId::new(2), 1)));
        assert!(!dynamic.contains(&(TensorId::new(2), 0)));
    }

    #[test]
    fn dynamic_axes_do_not_propagate_through_matmul_contraction() {
        let mut sample = GraphIr::new(vec![OperationIr::Float(
            burn::backend::DType::F32,
            FloatOperationIr::Matmul(MatmulOpIr {
                lhs: tensor(1, &[2, 3]),
                rhs: tensor(2, &[3, 5]),
                out: tensor(3, &[2, 5]),
            }),
        )]);
        sample.inputs = vec![TensorId::new(1), TensorId::new(2)];
        sample.outputs = vec![TensorId::new(3)];
        let mut validation = GraphIr::new(vec![OperationIr::Float(
            burn::backend::DType::F32,
            FloatOperationIr::Matmul(MatmulOpIr {
                lhs: tensor(11, &[2, 4]),
                rhs: tensor(12, &[4, 5]),
                out: tensor(13, &[2, 5]),
            }),
        )]);
        validation.inputs = vec![TensorId::new(11), TensorId::new(12)];
        validation.outputs = vec![TensorId::new(13)];
        let specs = [
            InputSpec::new([AxisSpec::Static, AxisSpec::dynamic("contract")]),
            InputSpec::new([AxisSpec::dynamic("contract"), AxisSpec::Static]),
        ];

        let dynamic = potentially_dynamic_axes(&sample, &validation, &specs);

        assert!(!dynamic.contains(&(TensorId::new(3), 0)));
        assert!(!dynamic.contains(&(TensorId::new(3), 1)));
    }

    #[test]
    fn dynamic_axes_do_not_mark_adaptive_pool_spatial_outputs() {
        let mut sample = GraphIr::new(vec![OperationIr::Module(
            ModuleOperationIr::AdaptiveAvgPool2d(AdaptiveAvgPool2dOpIr {
                x: tensor(1, &[1, 2, 4, 5]),
                output_size: [1, 1],
                out: tensor(2, &[1, 2, 1, 1]),
            }),
        )]);
        sample.inputs = vec![TensorId::new(1)];
        sample.outputs = vec![TensorId::new(2)];
        let mut validation = GraphIr::new(vec![OperationIr::Module(
            ModuleOperationIr::AdaptiveAvgPool2d(AdaptiveAvgPool2dOpIr {
                x: tensor(11, &[1, 2, 6, 7]),
                output_size: [1, 1],
                out: tensor(12, &[1, 2, 1, 1]),
            }),
        )]);
        validation.inputs = vec![TensorId::new(11)];
        validation.outputs = vec![TensorId::new(12)];
        let specs = [InputSpec::new([
            AxisSpec::Static,
            AxisSpec::Static,
            AxisSpec::dynamic("height"),
            AxisSpec::dynamic("width"),
        ])];

        let dynamic = potentially_dynamic_axes(&sample, &validation, &specs);

        assert!(!dynamic.contains(&(TensorId::new(2), 2)));
        assert!(!dynamic.contains(&(TensorId::new(2), 3)));
    }

    #[test]
    fn validator_rejects_static_attribute_changes() {
        let sample = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: tensor(1, &[2, 3]),
                out: tensor(2, &[3, 2]),
                dim1: 0,
                dim2: 1,
            },
        ))]);
        let validation = GraphIr::new(vec![OperationIr::BaseFloat(BaseOperationIr::SwapDims(
            SwapDimsOpIr {
                input: tensor(4, &[7, 3]),
                out: tensor(5, &[3, 7]),
                dim1: 1,
                dim2: 0,
            },
        ))]);
        assert!(matches!(
            GraphStructureValidator::validate(&sample, &validation),
            Err(ExportError::DynamicGraphMismatch { operation: 0, .. })
        ));
    }
}
