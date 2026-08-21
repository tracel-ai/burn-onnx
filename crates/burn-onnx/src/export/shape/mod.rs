mod dynamic_axes;

use burn::backend::ir::{
    BaseOperationIr, GraphIr, ModuleOperationIr, NumericOperationIr, OperationIr, TensorId,
    TensorIr,
};

use crate::export::{
    DynamicAxis, ExportError, GraphStructureValidator, ResolvedExportGraph, ResolvedShape,
    ShapeExpr,
};

use super::lower::patterns;
use dynamic_axes::PotentiallyDynamicAxes;

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

        let analysis = PairedTraceAnalysis::new(self)?;
        let dynamic_axes = analysis.dynamic_axes()?;
        let shapes = analysis.resolve_shapes()?;
        Ok(ResolvedExportGraph {
            graph: self.sample.clone(),
            shapes,
            dynamic_axes,
        })
    }
}

struct PairedInput<'a> {
    sample: &'a TensorIr,
    validation: &'a TensorIr,
    spec: &'a InputSpec,
}

enum DimensionResolution {
    Resolved(ShapeExpr),
    Infer,
}

impl DimensionResolution {
    fn is_inferred(&self) -> bool {
        matches!(self, Self::Infer)
    }

    fn into_expression(self) -> ShapeExpr {
        match self {
            Self::Resolved(expression) => expression,
            Self::Infer => ShapeExpr::Infer,
        }
    }
}

struct PairedTraceAnalysis<'a> {
    resolver: &'a PairedTraceShapeResolver<'a>,
    inputs: Vec<PairedInput<'a>>,
    potentially_dynamic: PotentiallyDynamicAxes,
}

impl<'a> PairedTraceAnalysis<'a> {
    fn new(resolver: &'a PairedTraceShapeResolver<'a>) -> Result<Self, ExportError> {
        let inputs = resolver
            .inputs
            .iter()
            .enumerate()
            .map(|(position, spec)| {
                let sample_id = resolver.sample.inputs[position];
                let validation_id = resolver.validation.inputs[position];
                Ok(PairedInput {
                    sample: tensor(resolver.sample, sample_id)
                        .ok_or(ExportError::MissingValue(sample_id))?,
                    validation: tensor(resolver.validation, validation_id)
                        .ok_or(ExportError::MissingValue(validation_id))?,
                    spec,
                })
            })
            .collect::<Result<_, _>>()?;
        let potentially_dynamic =
            PotentiallyDynamicAxes::analyze(resolver.sample, resolver.validation, resolver.inputs);
        Ok(Self {
            resolver,
            inputs,
            potentially_dynamic,
        })
    }

    fn dynamic_axes(&self) -> Result<Vec<DynamicAxis>, ExportError> {
        let mut axes = self.input_dynamic_axes();
        for (position, (&sample_id, &validation_id)) in self
            .resolver
            .sample
            .outputs
            .iter()
            .zip(&self.resolver.validation.outputs)
            .enumerate()
        {
            let sample = tensor(self.resolver.sample, sample_id)
                .ok_or(ExportError::MissingValue(sample_id))?;
            let validation = tensor(self.resolver.validation, validation_id)
                .ok_or(ExportError::MissingValue(validation_id))?;
            for axis in 0..sample.shape.num_dims().min(validation.shape.num_dims()) {
                if !self.potentially_dynamic.contains(sample_id, axis)
                    || axes
                        .iter()
                        .any(|dynamic| dynamic.tensor == sample_id && dynamic.axis == axis)
                {
                    continue;
                }
                axes.push(DynamicAxis {
                    tensor: sample_id,
                    axis,
                    symbol: self
                        .potentially_dynamic
                        .symbol(sample_id, axis)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("output_{position}_dim_{axis}")),
                });
            }
        }
        Ok(axes)
    }

    fn input_dynamic_axes(&self) -> Vec<DynamicAxis> {
        self.inputs
            .iter()
            .flat_map(|input| {
                input
                    .spec
                    .axes
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, spec)| match spec {
                        AxisSpec::Dynamic { symbol } => Some(DynamicAxis {
                            tensor: input.sample.id,
                            axis,
                            symbol: symbol.clone(),
                        }),
                        AxisSpec::Static => None,
                    })
            })
            .collect()
    }

    fn resolve_shapes(&self) -> Result<Vec<ResolvedShape>, ExportError> {
        let validation = shape_operations(self.resolver.validation).collect::<Vec<_>>();
        shape_operations(self.resolver.sample)
            .zip(validation)
            .map(|(sample, validation)| self.resolve_shape(sample, validation))
            .collect()
    }

    fn resolve_shape(
        &self,
        sample: ShapeOperation<'_>,
        validation: ShapeOperation<'_>,
    ) -> Result<ResolvedShape, ExportError> {
        let mut dimensions = Vec::new();
        let mut inferred_axes = Vec::new();
        for (axis, (&sample_dim, &validation_dim)) in sample
            .output
            .shape
            .iter()
            .zip(validation.output.shape.iter())
            .enumerate()
        {
            let resolution =
                self.resolve_dimension(&sample, &validation, axis, sample_dim, validation_dim)?;
            if resolution.is_inferred() {
                inferred_axes.push(axis);
            }
            dimensions.push(resolution.into_expression());
        }
        if inferred_axes.len() > 1 {
            return Err(ExportError::DynamicShapeLost {
                tensor: sample.output.id,
                axis: inferred_axes[0],
                reason: "multiple element-count-derived dimensions remain".into(),
            });
        }
        Ok(ResolvedShape {
            operation: sample.index,
            tensor: sample.output.id,
            dimensions,
        })
    }

    fn resolve_dimension(
        &self,
        sample: &ShapeOperation<'_>,
        validation: &ShapeOperation<'_>,
        axis: usize,
        sample_dim: usize,
        validation_dim: usize,
    ) -> Result<DimensionResolution, ExportError> {
        let mut candidates = self.input_dimension_candidates(sample_dim, validation_dim);
        if candidates.is_empty() {
            candidates.extend(self.source_dimension_candidates(
                sample,
                validation,
                sample_dim,
                validation_dim,
            ));
        }
        candidates.dedup();

        match candidates.as_slice() {
            [candidate] => Ok(DimensionResolution::Resolved(candidate.clone())),
            [] if sample_dim == validation_dim => {
                self.resolve_equal_dimension(sample, axis, sample_dim)
            }
            [] if sample.source.is_none() => Err(ExportError::DynamicShapeLost {
                tensor: sample.output.id,
                axis,
                reason: "Full dimension does not match an annotated dynamic input axis".into(),
            }),
            [] => Ok(DimensionResolution::Infer),
            candidates => Err(ExportError::DynamicShapeLost {
                tensor: sample.output.id,
                axis,
                reason: format!(
                    "dimension matches {} dynamic source axes and is ambiguous",
                    candidates.len()
                ),
            }),
        }
    }

    fn input_dimension_candidates(
        &self,
        sample_dim: usize,
        validation_dim: usize,
    ) -> Vec<ShapeExpr> {
        self.inputs
            .iter()
            .flat_map(|input| {
                input
                    .spec
                    .axes
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, spec)| {
                        (matches!(spec, AxisSpec::Dynamic { .. })
                            && input.sample.shape.get(axis) == Some(&sample_dim)
                            && input.validation.shape.get(axis) == Some(&validation_dim))
                        .then_some(ShapeExpr::InputDim {
                            input: input.sample.id,
                            axis,
                        })
                    })
            })
            .collect()
    }

    fn source_dimension_candidates(
        &self,
        sample: &ShapeOperation<'_>,
        validation: &ShapeOperation<'_>,
        sample_dim: usize,
        validation_dim: usize,
    ) -> Vec<ShapeExpr> {
        let (Some(source), Some(validation_source)) = (sample.source, validation.source) else {
            return Vec::new();
        };
        source
            .shape
            .iter()
            .zip(validation_source.shape.iter())
            .enumerate()
            .filter_map(|(axis, (&source_dim, &validation_source_dim))| {
                (self.potentially_dynamic.contains(source.id, axis)
                    && source_dim == sample_dim
                    && validation_source_dim == validation_dim)
                    .then_some(ShapeExpr::TensorDim {
                        tensor: source.id,
                        axis,
                    })
            })
            .collect()
    }

    fn resolve_equal_dimension(
        &self,
        operation: &ShapeOperation<'_>,
        axis: usize,
        dimension: usize,
    ) -> Result<DimensionResolution, ExportError> {
        if let Some(source) = operation.source {
            let has_static_match = source
                .shape
                .iter()
                .enumerate()
                .any(|(source_axis, &value)| {
                    value == dimension && !self.potentially_dynamic.contains(source.id, source_axis)
                });
            let has_dynamic_source =
                source.shape.iter().enumerate().any(|(source_axis, _)| {
                    self.potentially_dynamic.contains(source.id, source_axis)
                });
            return Ok(if has_static_match || !has_dynamic_source {
                DimensionResolution::Resolved(ShapeExpr::Static(dimension))
            } else {
                DimensionResolution::Infer
            });
        }

        if aligned_static_input_dimension(
            dimension,
            axis,
            operation.output.shape.num_dims(),
            self.resolver.sample,
            self.resolver.inputs,
        ) || !has_dynamic_inputs(self.resolver.inputs)
        {
            Ok(DimensionResolution::Resolved(ShapeExpr::Static(dimension)))
        } else {
            Err(ExportError::DynamicShapeLost {
                tensor: operation.output.id,
                axis,
                reason: "equal Full dimensions cannot be proven static from two captures".into(),
            })
        }
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
        let constant_pad = match (
            patterns::constant_pad(&sample.operations, index),
            patterns::constant_pad(&validation.operations, index),
        ) {
            (Some(sample_pad), Some(validation_pad)) if sample_pad.pads == validation_pad.pads => {
                true
            }
            (Some(sample_pad), Some(validation_pad)) => {
                let rank = sample_pad.slice_assign.out.shape.num_dims();
                let axis = sample_pad
                    .pads
                    .iter()
                    .zip(&validation_pad.pads)
                    .position(|(sample, validation)| sample != validation)
                    .map(|position| position % rank)
                    .unwrap_or(0);
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample_pad.slice_assign.out.id,
                    axis,
                    reason: format!("operation {index} has varying constant-padding widths"),
                });
            }
            (Some(sample_pad), None) => {
                return Err(ExportError::DynamicShapeLost {
                    tensor: sample_pad.slice_assign.out.id,
                    axis: 0,
                    reason: format!(
                        "operation {index} is recognized as constant padding in only one trace"
                    ),
                });
            }
            (None, Some(validation_pad)) => {
                return Err(ExportError::DynamicShapeLost {
                    tensor: validation_pad.slice_assign.out.id,
                    axis: 0,
                    reason: format!(
                        "operation {index} is recognized as constant padding in only one trace"
                    ),
                });
            }
            (None, None) => false,
        };
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
                if constant_pad {
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
    source: Option<&'a TensorIr>,
    output: &'a TensorIr,
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

fn tensor(graph: &GraphIr, id: TensorId) -> Option<&TensorIr> {
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
        FullOpIr, InterpolateModeIr, InterpolateOpIr, InterpolateOptionsIr, ScalarIr, ShapeOpIr,
        SwapDimsOpIr, TensorIr,
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
