//! Dynamic-axis dependency analysis for paired captures.

use burn::backend::ir::{
    ActivationOperationIr, BaseOperationIr, FloatOperationIr, GraphIr, IntOperationIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, TensorId, TensorIr,
};
use hashbrown::HashMap;

use super::{AxisSpec, InputSpec};

/// Axes whose dimensions may depend on a declared dynamic input.
pub(super) struct PotentiallyDynamicAxes {
    axes: HashMap<(TensorId, usize), AxisDependency>,
}

impl PotentiallyDynamicAxes {
    /// Analyze two structurally equivalent captures.
    pub(super) fn analyze(sample: &GraphIr, validation: &GraphIr, specs: &[InputSpec]) -> Self {
        let mut tracker = AxisTracker::from_inputs(sample, specs);
        for (sample_operation, validation_operation) in
            sample.operations.iter().zip(&validation.operations)
        {
            tracker.transfer(sample_operation);
            tracker.transfer_reshape(sample_operation, validation_operation);
            tracker.observe_changes(sample_operation, validation_operation);
        }
        Self { axes: tracker.axes }
    }

    /// Whether one tensor axis can vary at runtime.
    pub(super) fn contains(&self, tensor: TensorId, axis: usize) -> bool {
        self.axes.contains_key(&(tensor, axis))
    }

    /// Input symbol preserved exactly by one output axis, if known.
    pub(super) fn symbol(&self, tensor: TensorId, axis: usize) -> Option<&str> {
        match self.axes.get(&(tensor, axis)) {
            Some(AxisDependency::Exact(symbol)) => Some(symbol),
            Some(AxisDependency::Derived) | None => None,
        }
    }
}

#[derive(Clone)]
enum AxisDependency {
    Exact(String),
    Derived,
}

/// Mutable dependency set with reusable shape-transfer operations.
struct AxisTracker {
    axes: HashMap<(TensorId, usize), AxisDependency>,
}

impl AxisTracker {
    fn from_inputs(graph: &GraphIr, specs: &[InputSpec]) -> Self {
        let mut axes = HashMap::new();
        for (position, spec) in specs.iter().enumerate() {
            let Some(&input) = graph.inputs.get(position) else {
                continue;
            };
            for (axis, axis_spec) in spec.axes.iter().enumerate() {
                if let AxisSpec::Dynamic { symbol } = axis_spec {
                    axes.insert((input, axis), AxisDependency::Exact(symbol.clone()));
                }
            }
        }
        Self { axes }
    }

    fn contains(&self, tensor: &TensorIr, axis: usize) -> bool {
        self.axes.contains_key(&(tensor.id, axis))
    }

    fn mark_derived(&mut self, tensor: &TensorIr, axis: usize) {
        self.axes.insert((tensor.id, axis), AxisDependency::Derived);
    }

    fn merge(&mut self, tensor: &TensorIr, axis: usize, dependency: AxisDependency) {
        self.axes
            .entry((tensor.id, axis))
            .and_modify(|current| {
                if !matches!(
                    (&*current, &dependency),
                    (AxisDependency::Exact(current), AxisDependency::Exact(next))
                        if current == next
                ) {
                    *current = AxisDependency::Derived;
                }
            })
            .or_insert(dependency);
    }

    fn observe_changes(&mut self, sample: &OperationIr, validation: &OperationIr) {
        for (sample_output, validation_output) in sample.outputs().zip(validation.outputs()) {
            for (axis, (&sample_dim, &validation_dim)) in sample_output
                .shape
                .iter()
                .zip(validation_output.shape.iter())
                .enumerate()
            {
                if sample_dim != validation_dim {
                    self.axes
                        .entry((sample_output.id, axis))
                        .or_insert(AxisDependency::Derived);
                }
            }
        }
    }

    fn transfer(&mut self, operation: &OperationIr) {
        match operation {
            OperationIr::BaseFloat(operation)
            | OperationIr::BaseInt(operation)
            | OperationIr::BaseBool(operation) => self.transfer_base(operation),
            OperationIr::NumericFloat(_, operation) | OperationIr::NumericInt(_, operation) => {
                self.transfer_numeric(operation)
            }
            OperationIr::Float(_, operation) => self.transfer_float(operation),
            OperationIr::Int(IntOperationIr::Matmul(operation)) => {
                self.matmul(&operation.lhs, &operation.rhs, &operation.out)
            }
            OperationIr::Activation(
                ActivationOperationIr::Relu(operation) | ActivationOperationIr::Sigmoid(operation),
            ) => self.same_axes(&operation.input, &operation.out),
            OperationIr::Module(operation) => self.transfer_module(operation),
            _ => {}
        }
    }

    fn transfer_base(&mut self, operation: &BaseOperationIr) {
        match operation {
            BaseOperationIr::SwapDims(operation) => {
                for output_axis in 0..operation.out.shape.num_dims() {
                    let input_axis = match output_axis {
                        axis if axis == operation.dim1 => operation.dim2,
                        axis if axis == operation.dim2 => operation.dim1,
                        axis => axis,
                    };
                    self.axis(&operation.input, input_axis, &operation.out, output_axis);
                }
            }
            BaseOperationIr::Permute(operation) => {
                for (output_axis, &input_axis) in operation.axes.iter().enumerate() {
                    self.axis(&operation.input, input_axis, &operation.out, output_axis);
                }
            }
            BaseOperationIr::Flip(operation) => self.same_axes(&operation.input, &operation.out),
            BaseOperationIr::Expand(operation) => {
                self.broadcast_axes(&operation.input, &operation.out)
            }
            BaseOperationIr::RepeatDim(operation) => {
                for axis in 0..operation.out.shape.num_dims() {
                    if axis == operation.dim {
                        self.derived_axis(&operation.tensor, axis, &operation.out, axis);
                    } else {
                        self.axis(&operation.tensor, axis, &operation.out, axis);
                    }
                }
            }
            BaseOperationIr::Cat(operation) => {
                for input in &operation.tensors {
                    for axis in 0..operation.out.shape.num_dims() {
                        if axis == operation.dim {
                            self.derived_axis(input, axis, &operation.out, axis);
                        } else {
                            self.axis(input, axis, &operation.out, axis);
                        }
                    }
                }
            }
            BaseOperationIr::Cast(operation) => self.same_axes(&operation.input, &operation.out),
            BaseOperationIr::AllDim(operation) | BaseOperationIr::AnyDim(operation) => {
                self.reduced_axes(&operation.input, &operation.out, operation.axis)
            }
            // Reshape has a value-aware transfer after the general operation transfer.
            _ => {}
        }
    }

    fn transfer_numeric(&mut self, operation: &NumericOperationIr) {
        match operation {
            NumericOperationIr::Add(operation)
            | NumericOperationIr::Sub(operation)
            | NumericOperationIr::Mul(operation)
            | NumericOperationIr::Div(operation) => {
                self.broadcast_axes(&operation.lhs, &operation.out);
                self.broadcast_axes(&operation.rhs, &operation.out);
            }
            NumericOperationIr::AddScalar(operation)
            | NumericOperationIr::SubScalar(operation)
            | NumericOperationIr::MulScalar(operation)
            | NumericOperationIr::DivScalar(operation) => {
                self.same_axes(&operation.lhs, &operation.out)
            }
            NumericOperationIr::Abs(operation) | NumericOperationIr::Neg(operation) => {
                self.same_axes(&operation.input, &operation.out)
            }
            NumericOperationIr::Pad(operation) => {
                for (axis, &(before, after)) in operation.padding.iter().enumerate() {
                    if before == 0 && after == 0 {
                        self.axis(&operation.input, axis, &operation.out, axis);
                    } else {
                        self.derived_axis(&operation.input, axis, &operation.out, axis);
                    }
                }
            }
            NumericOperationIr::MeanDim(operation)
            | NumericOperationIr::SumDim(operation)
            | NumericOperationIr::ProdDim(operation) => {
                self.reduced_axes(&operation.input, &operation.out, operation.axis)
            }
            _ => {}
        }
    }

    fn transfer_float(&mut self, operation: &FloatOperationIr) {
        match operation {
            FloatOperationIr::Exp(operation)
            | FloatOperationIr::Log(operation)
            | FloatOperationIr::Sqrt(operation)
            | FloatOperationIr::Tanh(operation) => self.same_axes(&operation.input, &operation.out),
            FloatOperationIr::Matmul(operation) => {
                self.matmul(&operation.lhs, &operation.rhs, &operation.out)
            }
            _ => {}
        }
    }

    fn transfer_module(&mut self, operation: &ModuleOperationIr) {
        match operation {
            ModuleOperationIr::Conv2d(operation) => {
                self.axis(&operation.x, 0, &operation.out, 0);
                self.derived_axis(&operation.x, 2, &operation.out, 2);
                self.derived_axis(&operation.x, 3, &operation.out, 3);
            }
            ModuleOperationIr::BatchNorm(operation) => self.same_axes(&operation.x, &operation.out),
            ModuleOperationIr::Interpolate(operation) => {
                self.leading_axes(&operation.x, &operation.out, 2)
            }
            ModuleOperationIr::AdaptiveAvgPool2d(operation) => {
                self.leading_axes(&operation.x, &operation.out, 2)
            }
            ModuleOperationIr::MaxPool2d(operation) => {
                self.leading_axes(&operation.x, &operation.out, 2);
                self.derived_axis(&operation.x, 2, &operation.out, 2);
                self.derived_axis(&operation.x, 3, &operation.out, 3);
            }
            ModuleOperationIr::AvgPool2d(operation) => {
                self.leading_axes(&operation.x, &operation.out, 2);
                self.derived_axis(&operation.x, 2, &operation.out, 2);
                self.derived_axis(&operation.x, 3, &operation.out, 3);
            }
            ModuleOperationIr::Linear(operation) => self.leading_axes(
                &operation.x,
                &operation.out,
                operation.out.shape.num_dims().saturating_sub(1),
            ),
            _ => {}
        }
    }

    fn transfer_reshape(&mut self, sample: &OperationIr, validation: &OperationIr) {
        let (Some((source, output)), Some((validation_source, validation_output))) =
            (reshape_tensors(sample), reshape_tensors(validation))
        else {
            return;
        };
        let source_has_dynamic = source
            .shape
            .iter()
            .enumerate()
            .any(|(axis, _)| self.contains(source, axis));
        for (output_axis, (&output_dim, &validation_output_dim)) in output
            .shape
            .iter()
            .zip(validation_output.shape.iter())
            .enumerate()
        {
            if self.contains(output, output_axis) {
                continue;
            }
            let dynamic_source_matches = source
                .shape
                .iter()
                .zip(validation_source.shape.iter())
                .enumerate()
                .filter_map(|(source_axis, (&source_dim, &validation_source_dim))| {
                    self.contains(source, source_axis).then_some((
                        source_axis,
                        source_dim,
                        validation_source_dim,
                    ))
                })
                .filter(|(_, source_dim, validation_source_dim)| {
                    *source_dim == output_dim && *validation_source_dim == validation_output_dim
                })
                .map(|(source_axis, _, _)| source_axis)
                .collect::<Vec<_>>();
            let static_source_match =
                source
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(source_axis, &source_dim)| {
                        !self.contains(source, source_axis) && source_dim == output_dim
                    });
            if dynamic_source_matches.is_empty() {
                if source_has_dynamic && !static_source_match {
                    self.mark_derived(output, output_axis);
                }
            } else {
                for source_axis in dynamic_source_matches {
                    self.axis(source, source_axis, output, output_axis);
                }
            }
        }
    }

    fn axis(&mut self, input: &TensorIr, input_axis: usize, output: &TensorIr, output_axis: usize) {
        if let Some(dependency) = self.axes.get(&(input.id, input_axis)).cloned() {
            self.merge(output, output_axis, dependency);
        }
    }

    fn derived_axis(
        &mut self,
        input: &TensorIr,
        input_axis: usize,
        output: &TensorIr,
        output_axis: usize,
    ) {
        if self.contains(input, input_axis) {
            self.mark_derived(output, output_axis);
        }
    }

    fn same_axes(&mut self, input: &TensorIr, output: &TensorIr) {
        self.leading_axes(
            input,
            output,
            input.shape.num_dims().min(output.shape.num_dims()),
        );
    }

    fn leading_axes(&mut self, input: &TensorIr, output: &TensorIr, count: usize) {
        for axis in 0..count {
            self.axis(input, axis, output, axis);
        }
    }

    fn broadcast_axes(&mut self, input: &TensorIr, output: &TensorIr) {
        let Some(offset) = output.shape.num_dims().checked_sub(input.shape.num_dims()) else {
            return;
        };
        for input_axis in 0..input.shape.num_dims() {
            self.axis(input, input_axis, output, offset + input_axis);
        }
    }

    fn reduced_axes(&mut self, input: &TensorIr, output: &TensorIr, reduced_axis: usize) {
        for axis in 0..input.shape.num_dims().min(output.shape.num_dims()) {
            if axis != reduced_axis {
                self.axis(input, axis, output, axis);
            }
        }
    }

    fn matmul(&mut self, lhs: &TensorIr, rhs: &TensorIr, output: &TensorIr) {
        let output_rank = output.shape.num_dims();
        let lhs_rank = lhs.shape.num_dims();
        let rhs_rank = rhs.shape.num_dims();
        if output_rank < 2 || lhs_rank < 2 || rhs_rank < 2 {
            return;
        }

        self.batch_axes(lhs, output, lhs_rank - 2, output_rank - 2);
        self.batch_axes(rhs, output, rhs_rank - 2, output_rank - 2);
        self.axis(lhs, lhs_rank - 2, output, output_rank - 2);
        self.axis(rhs, rhs_rank - 1, output, output_rank - 1);
    }

    fn batch_axes(
        &mut self,
        input: &TensorIr,
        output: &TensorIr,
        input_count: usize,
        output_count: usize,
    ) {
        for input_axis in 0..input_count {
            self.axis(
                input,
                input_axis,
                output,
                output_count - input_count + input_axis,
            );
        }
    }
}

fn reshape_tensors(operation: &OperationIr) -> Option<(&TensorIr, &TensorIr)> {
    match operation {
        OperationIr::BaseFloat(BaseOperationIr::Reshape(operation))
        | OperationIr::BaseInt(BaseOperationIr::Reshape(operation))
        | OperationIr::BaseBool(BaseOperationIr::Reshape(operation)) => {
            Some((&operation.input, &operation.out))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ir::{AdaptiveAvgPool2dOpIr, MatmulOpIr, SwapDimsOpIr};
    use burn::backend::{DType, Shape};

    fn tensor(id: u64, shape: &[usize]) -> TensorIr {
        TensorIr::uninit(
            TensorId::new(id),
            shape.iter().copied().collect::<Shape>(),
            DType::F32,
        )
    }

    fn graph(operation: OperationIr, inputs: &[u64], outputs: &[u64]) -> GraphIr {
        let mut graph = GraphIr::new(vec![operation]);
        graph.inputs = inputs.iter().copied().map(TensorId::new).collect();
        graph.outputs = outputs.iter().copied().map(TensorId::new).collect();
        graph
    }

    #[test]
    fn follows_swapped_axis_positions_without_marking_the_old_position() {
        let sample = graph(
            OperationIr::BaseFloat(BaseOperationIr::SwapDims(SwapDimsOpIr {
                input: tensor(1, &[2, 3]),
                out: tensor(2, &[3, 2]),
                dim1: 0,
                dim2: 1,
            })),
            &[1],
            &[2],
        );
        let validation = graph(
            OperationIr::BaseFloat(BaseOperationIr::SwapDims(SwapDimsOpIr {
                input: tensor(11, &[4, 3]),
                out: tensor(12, &[3, 4]),
                dim1: 0,
                dim2: 1,
            })),
            &[11],
            &[12],
        );
        let specs = [InputSpec::new([
            AxisSpec::dynamic("rows"),
            AxisSpec::Static,
        ])];

        let dynamic = PotentiallyDynamicAxes::analyze(&sample, &validation, &specs);

        assert!(dynamic.contains(TensorId::new(2), 1));
        assert!(!dynamic.contains(TensorId::new(2), 0));
    }

    #[test]
    fn does_not_propagate_through_matmul_contraction() {
        let sample = graph(
            OperationIr::Float(
                DType::F32,
                FloatOperationIr::Matmul(MatmulOpIr {
                    lhs: tensor(1, &[2, 3]),
                    rhs: tensor(2, &[3, 5]),
                    out: tensor(3, &[2, 5]),
                }),
            ),
            &[1, 2],
            &[3],
        );
        let validation = graph(
            OperationIr::Float(
                DType::F32,
                FloatOperationIr::Matmul(MatmulOpIr {
                    lhs: tensor(11, &[2, 4]),
                    rhs: tensor(12, &[4, 5]),
                    out: tensor(13, &[2, 5]),
                }),
            ),
            &[11, 12],
            &[13],
        );
        let specs = [
            InputSpec::new([AxisSpec::Static, AxisSpec::dynamic("contract")]),
            InputSpec::new([AxisSpec::dynamic("contract"), AxisSpec::Static]),
        ];

        let dynamic = PotentiallyDynamicAxes::analyze(&sample, &validation, &specs);

        assert!(!dynamic.contains(TensorId::new(3), 0));
        assert!(!dynamic.contains(TensorId::new(3), 1));
    }

    #[test]
    fn does_not_mark_adaptive_pool_spatial_outputs() {
        let sample = graph(
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(
                AdaptiveAvgPool2dOpIr {
                    x: tensor(1, &[1, 2, 4, 5]),
                    output_size: [1, 1],
                    out: tensor(2, &[1, 2, 1, 1]),
                },
            )),
            &[1],
            &[2],
        );
        let validation = graph(
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(
                AdaptiveAvgPool2dOpIr {
                    x: tensor(11, &[1, 2, 6, 7]),
                    output_size: [1, 1],
                    out: tensor(12, &[1, 2, 1, 1]),
                },
            )),
            &[11],
            &[12],
        );
        let specs = [InputSpec::new([
            AxisSpec::Static,
            AxisSpec::Static,
            AxisSpec::dynamic("height"),
            AxisSpec::dynamic("width"),
        ])];

        let dynamic = PotentiallyDynamicAxes::analyze(&sample, &validation, &specs);

        assert!(!dynamic.contains(TensorId::new(2), 2));
        assert!(!dynamic.contains(TensorId::new(2), 3));
    }
}
