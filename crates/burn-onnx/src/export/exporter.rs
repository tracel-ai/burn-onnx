use std::{string::String, vec::Vec};

use burn::backend::capture::{CaptureBackend, CapturedGraph, TensorId};
use burn::backend::ir::OperationIr;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Bool, Device, Float, Int, Tensor};
use hashbrown::{HashMap, HashSet};

use crate::export::{
    ExportError, InputSpec, OnnxModel,
    lower::export_graph_with_bindings_and_opset,
    shape::{PairedTraceShapeResolver, ShapeResolver, StaticShapeResolver, validate_input_specs},
};

/// ONNX operator-set versions supported by the exporter.
///
/// Operator schemas and attributes change between operator sets, so this is a
/// capability selection rather than an arbitrary integer written to the model.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Opset {
    /// ONNX operator set 18.
    #[default]
    V18,
}

impl Opset {
    /// Return the numeric ONNX operator-set version.
    pub const fn version(self) -> i64 {
        match self {
            Self::V18 => 18,
        }
    }
}

mod sealed {
    use super::*;

    #[doc(hidden)]
    pub trait SealedExportOutput {
        fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) -> Result<(), ExportError>;

        fn tensor_ids(&self) -> Result<Vec<TensorId>, ExportError> {
            let mut ids = Vec::new();
            self.collect_tensor_ids(&mut ids)?;
            Ok(ids)
        }
    }

    #[doc(hidden)]
    pub trait SealedExportInput: SealedExportOutput + Sized {
        fn collect_input_shapes(&self, shapes: &mut Vec<Vec<usize>>);

        fn to_capture_device(self, device: &Device) -> Self;

        fn input_shapes(&self) -> Vec<Vec<usize>> {
            let mut shapes = Vec::new();
            self.collect_input_shapes(&mut shapes);
            shapes
        }
    }
}

/// Values returned by an exported forward function.
///
/// This trait is sealed. The exporter supports Burn tensors, vectors, and
/// tuples for which this crate provides implementations.
pub trait ExportOutput: sealed::SealedExportOutput {}

impl<T: sealed::SealedExportOutput> ExportOutput for T {}

/// Runtime input values which can be moved to the private capture device.
///
/// This trait is sealed to the input forms supported by the exporter.
pub trait ExportInput: ExportOutput + sealed::SealedExportInput {}

impl<T: sealed::SealedExportInput> ExportInput for T {}

macro_rules! impl_tensor_value {
    ($kind:ty) => {
        impl<const D: usize> sealed::SealedExportOutput for Tensor<D, $kind> {
            fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) -> Result<(), ExportError> {
                let primitive = self
                    .clone()
                    .try_into_primitive::<CaptureBackend>()
                    .map_err(|error| {
                        ExportError::InvalidBoundary(format!(
                            "export tensor must be on the capture device: {error:?}"
                        ))
                    })?;
                ids.push(primitive.id());
                Ok(())
            }
        }

        impl<const D: usize> sealed::SealedExportInput for Tensor<D, $kind> {
            fn collect_input_shapes(&self, shapes: &mut Vec<Vec<usize>>) {
                shapes.push(self.dims().to_vec());
            }

            fn to_capture_device(self, device: &Device) -> Self {
                self.to_device(device)
            }
        }
    };
}

impl_tensor_value!(Float);
impl_tensor_value!(Int);
impl_tensor_value!(Bool);

impl<T: ExportOutput> sealed::SealedExportOutput for Vec<T> {
    fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) -> Result<(), ExportError> {
        for value in self {
            value.collect_tensor_ids(ids)?;
        }
        Ok(())
    }
}

impl<T: ExportInput> sealed::SealedExportInput for Vec<T> {
    fn collect_input_shapes(&self, shapes: &mut Vec<Vec<usize>>) {
        for value in self {
            value.collect_input_shapes(shapes);
        }
    }

    fn to_capture_device(self, device: &Device) -> Self {
        self.into_iter()
            .map(|value| value.to_capture_device(device))
            .collect()
    }
}

macro_rules! impl_export_tuple {
    ($($name:ident),+) => {
        impl<$($name: ExportOutput),+> sealed::SealedExportOutput for ($($name,)+) {
            #[allow(non_snake_case)]
            fn collect_tensor_ids(&self, ids: &mut Vec<TensorId>) -> Result<(), ExportError> {
                let ($($name,)+) = self;
                $($name.collect_tensor_ids(ids)?;)+
                Ok(())
            }
        }
        impl<$($name: ExportInput),+> sealed::SealedExportInput for ($($name,)+) {
            #[allow(non_snake_case)]
            fn collect_input_shapes(&self, shapes: &mut Vec<Vec<usize>>) {
                let ($($name,)+) = self;
                $($name.collect_input_shapes(shapes);)+
            }

            #[allow(non_snake_case)]
            fn to_capture_device(self, device: &Device) -> Self {
                let ($($name,)+) = self;
                ($($name.to_capture_device(device),)+)
            }
        }
    };
}

impl_export_tuple!(A, B);
impl_export_tuple!(A, B, C);
impl_export_tuple!(A, B, C, D);

struct CapturedForward {
    captured: CapturedGraph,
    input_ids: Vec<TensorId>,
    parameter_names: HashMap<TensorId, String>,
}

#[derive(Default)]
struct ParameterNameVisitor {
    path: Vec<String>,
    names: HashMap<TensorId, String>,
    error: Option<ExportError>,
}

impl ParameterNameVisitor {
    fn record(&mut self, id: TensorId) {
        self.names.entry(id).or_insert_with(|| self.path.join("."));
    }

    fn record_error(&mut self, error: impl core::fmt::Debug) {
        if self.error.is_none() {
            self.error = Some(ExportError::InvalidBoundary(format!(
                "module parameter must be on the capture device: {error:?}"
            )));
        }
    }
}

impl ModuleVisitor for ParameterNameVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.into());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        match param.val().try_into_primitive::<CaptureBackend>() {
            Ok(tensor) => self.record(tensor.id()),
            Err(error) => self.record_error(error),
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        match param.val().try_into_primitive::<CaptureBackend>() {
            Ok(tensor) => self.record(tensor.id()),
            Err(error) => self.record_error(error),
        }
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        match param.val().try_into_primitive::<CaptureBackend>() {
            Ok(tensor) => self.record(tensor.id()),
            Err(error) => self.record_error(error),
        }
    }
}

/// High-level forward-capture ONNX exporter.
///
/// The exporter clones an ordinary Burn module and moves both it and the sample
/// inputs onto a private capture device. It then invokes the supplied forward
/// closure, identifies runtime boundaries, classifies other initialized values
/// as parameters, and emits an embedded-weight ONNX protobuf.
#[derive(Clone, Debug, Default)]
pub struct OnnxExporter {
    opset: Opset,
}

impl OnnxExporter {
    /// Create an ONNX exporter.
    pub const fn new() -> Self {
        Self { opset: Opset::V18 }
    }

    /// Select the ONNX operator set used for lowering.
    pub const fn opset(mut self, opset: Opset) -> Self {
        self.opset = opset;
        self
    }

    /// Return the selected ONNX operator set.
    pub const fn selected_opset(&self) -> Opset {
        self.opset
    }

    /// Capture one module forward pass and return a static-shape ONNX model.
    pub fn export<M, I, O, F>(
        &self,
        module: &M,
        inputs: I,
        forward: F,
    ) -> Result<OnnxModel, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportOutput,
        F: FnOnce(&M, I) -> O,
    {
        let captured = self.capture_forward(module, inputs, forward)?;
        let resolved = StaticShapeResolver {
            graph: &captured.captured.graph,
        }
        .resolve()?;
        export_graph_with_bindings_and_opset(
            &resolved,
            &captured.captured.values,
            &captured.input_ids,
            &captured.parameter_names,
            self.opset,
        )
    }

    /// Capture two shapes, validate their structure, and export symbolic input axes.
    ///
    /// `input_specs` are positional and must contain one entry per tensor in
    /// `sample_inputs`. Static axes must agree between both input sets; dynamic
    /// axes must differ. Repeated symbols must refer to identical dimensions.
    pub fn export_dynamic<M, I, O, F>(
        &self,
        module: &M,
        sample_inputs: I,
        validation_inputs: I,
        input_specs: &[InputSpec],
        forward: F,
    ) -> Result<OnnxModel, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportOutput,
        F: Fn(&M, I) -> O,
    {
        let sample_shapes = sample_inputs.input_shapes();
        let validation_shapes = validation_inputs.input_shapes();
        validate_input_specs(input_specs, &sample_shapes, &validation_shapes)?;
        let sample = self.capture_forward(module, sample_inputs, &forward)?;
        let validation = self.capture_forward(module, validation_inputs, &forward)?;
        let resolved = PairedTraceShapeResolver {
            sample: &sample.captured.graph,
            validation: &validation.captured.graph,
            inputs: input_specs,
        }
        .resolve()?;
        export_graph_with_bindings_and_opset(
            &resolved,
            &sample.captured.values,
            &sample.input_ids,
            &sample.parameter_names,
            self.opset,
        )
    }

    fn capture_forward<M, I, O, F>(
        &self,
        module: &M,
        inputs: I,
        forward: F,
    ) -> Result<CapturedForward, ExportError>
    where
        M: Module,
        I: ExportInput,
        O: ExportOutput,
        F: FnOnce(&M, I) -> O,
    {
        // Capture clients are scope-local. A distinct logical device prevents multi-backend
        // bridge caches from returning tensors owned by an earlier, already-closed scope.
        let device = Device::capture();
        let mut capture_metadata =
            None::<Result<(Vec<TensorId>, HashMap<TensorId, String>), ExportError>>;
        let captured = device.capture_scope(|scope| {
            // Module parameters, runtime inputs, and the forward pass must all use the client
            // installed for this scope. Operations outside the scope have no capture session.
            let module = module.clone().to_device(&device);
            let mut visitor = ParameterNameVisitor::default();
            module.visit(&mut visitor);
            if let Some(error) = visitor.error {
                capture_metadata = Some(Err(error));
                return scope.complete([], []);
            }
            let inputs = inputs.to_capture_device(&device);
            let input_ids = match inputs.tensor_ids() {
                Ok(ids) => ids,
                Err(error) => {
                    capture_metadata = Some(Err(error));
                    return scope.complete([], []);
                }
            };
            let output_ids = match forward(&module, inputs).tensor_ids() {
                Ok(ids) => ids,
                Err(error) => {
                    capture_metadata = Some(Err(error));
                    return scope.complete([], []);
                }
            };

            capture_metadata = Some(Ok((input_ids.clone(), visitor.names)));
            scope.complete(input_ids, output_ids)
        });
        let Some(capture_metadata) = capture_metadata else {
            return Err(ExportError::InvalidBoundary(
                "capture scope did not execute".into(),
            ));
        };
        let (input_ids, parameter_names) = capture_metadata?;
        let mut captured =
            captured.map_err(|error| ExportError::InvalidBoundary(error.to_string()))?;
        let boundaries = captured
            .graph
            .inputs
            .iter()
            .chain(&captured.graph.outputs)
            .copied()
            .collect::<HashSet<_>>();
        captured
            .graph
            .operations
            .retain(|operation| match operation {
                OperationIr::Init(operation) => boundaries.contains(&operation.out.id),
                _ => true,
            });
        validate_capture(&captured, &input_ids, &parameter_names)?;
        Ok(CapturedForward {
            captured,
            input_ids,
            parameter_names,
        })
    }
}

fn validate_capture(
    captured: &CapturedGraph,
    runtime_inputs: &[TensorId],
    parameter_names: &HashMap<TensorId, String>,
) -> Result<(), ExportError> {
    for (kind, boundaries) in [
        ("input", captured.graph.inputs.as_slice()),
        ("output", captured.graph.outputs.as_slice()),
    ] {
        let mut unique = HashSet::new();
        if let Some(id) = boundaries.iter().find(|id| !unique.insert(**id)) {
            return Err(ExportError::InvalidBoundary(format!(
                "duplicate graph {kind} tensor {id}"
            )));
        }
    }
    if captured.graph.inputs != runtime_inputs {
        return Err(ExportError::InvalidBoundary(
            "captured graph inputs do not match runtime input declaration order".into(),
        ));
    }
    for &id in runtime_inputs {
        if !captured.values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
    }
    for &id in parameter_names.keys() {
        if !captured.values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
    }

    let mut known: HashSet<_> = captured.values.keys().copied().collect();
    let mut metadata = HashMap::new();
    for (index, operation) in captured.graph.operations.iter().enumerate() {
        for tensor in operation.inputs() {
            metadata.entry(tensor.id).or_insert(tensor);
            if !known.contains(&tensor.id) {
                return Err(ExportError::InvalidBoundary(format!(
                    "operation {index} reads tensor {} before it is initialized or produced",
                    tensor.id
                )));
            }
        }
        for tensor in operation.outputs() {
            metadata.entry(tensor.id).or_insert(tensor);
            known.insert(tensor.id);
        }
    }
    for &id in &captured.graph.outputs {
        if !known.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "graph output tensor {id} is not initialized or produced"
            )));
        }
    }
    for (&id, data) in &captured.values {
        let Some(tensor) = metadata.get(&id) else {
            continue;
        };
        if tensor.dtype != data.dtype || tensor.shape != data.shape {
            return Err(ExportError::InvalidValue {
                tensor: id,
                reason: format!(
                    "captured metadata is {:?} {:?}, initialized value is {:?} {:?}",
                    tensor.dtype, tensor.shape, data.dtype, data.shape
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::nn::{Linear, LinearConfig, Relu};
    use core::cell::Cell;
    use onnx_ir::ModelProto;
    use protobuf::Message;

    #[derive(Module, Debug)]
    struct AddModule {
        weight: Param<Tensor<1>>,
    }

    #[derive(Module, Debug)]
    struct Mlp {
        first: Linear,
        activation: Relu,
        second: Linear,
    }

    #[test]
    fn exporter_uses_typed_opset_configuration() {
        let exporter = OnnxExporter::new().opset(Opset::V18);

        assert_eq!(exporter.selected_opset(), Opset::V18);
        assert_eq!(
            exporter.selected_opset().version(),
            Opset::default().version()
        );
    }

    #[test]
    fn captures_forward_and_embeds_module_value() {
        let device = Device::default();
        let exporter = OnnxExporter::new();
        let module = AddModule {
            weight: Param::from_data([2.0f32, 3.0], &device),
        };
        let input = Tensor::<1>::from_floats([5.0f32, 7.0], &device);

        let bytes = exporter
            .export(&module, input, |module, input| input + module.weight.val())
            .unwrap();
        let model = ModelProto::parse_from_bytes(bytes.as_bytes()).unwrap();
        assert_eq!(model.graph.node[0].op_type, "Add");
        assert_eq!(model.graph.input.len(), 1);
        assert_eq!(model.graph.output.len(), 1);
        assert_eq!(model.graph.initializer.len(), 1);
        assert_eq!(model.graph.initializer[0].name, "weight");
        assert_eq!(
            model.graph.initializer[0].raw_data.len(),
            2 * size_of::<f32>()
        );
    }

    #[test]
    fn saves_exported_model() {
        let device = Device::default();
        let module = AddModule {
            weight: Param::from_data([2.0f32, 3.0], &device),
        };
        let input = Tensor::<1>::from_floats([5.0f32, 7.0], &device);
        let model = OnnxExporter::new()
            .export(&module, input, |module, input| input + module.weight.val())
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");

        model.save(&path).unwrap();

        assert_eq!(std::fs::read(path).unwrap(), model.as_bytes());
        ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
    }

    #[test]
    fn exports_two_layer_mlp_forward() {
        let device = Device::default();
        let module = Mlp {
            first: LinearConfig::new(4, 3).init(&device),
            activation: Relu::new(),
            second: LinearConfig::new(3, 2).init(&device),
        };
        let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

        let bytes = OnnxExporter::new()
            .export(&module, input, |module, input| {
                let hidden = module.first.forward(input);
                module.second.forward(module.activation.forward(hidden))
            })
            .unwrap();
        let model = ModelProto::parse_from_bytes(bytes.as_bytes()).unwrap();
        let operations: Vec<_> = model
            .graph
            .node
            .iter()
            .map(|node| node.op_type.as_str())
            .collect();
        assert_eq!(
            operations,
            [
                "MatMul", "Reshape", "Add", "Relu", "MatMul", "Reshape", "Add"
            ]
        );
        // Four module parameters plus two constant bias-reshape operands.
        assert_eq!(model.graph.initializer.len(), 6);
        let initializer_names = model
            .graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert!(initializer_names.contains(&"first.weight"));
        assert!(initializer_names.contains(&"first.bias"));
        assert!(initializer_names.contains(&"second.weight"));
        assert!(initializer_names.contains(&"second.bias"));
        assert_eq!(model.graph.input.len(), 1);
        assert_eq!(model.graph.output.len(), 1);
    }

    #[test]
    fn dynamic_specs_are_validated_before_forward() {
        let device = Device::default();
        let module = AddModule {
            weight: Param::from_data([2.0f32, 3.0], &device),
        };
        let sample = Tensor::<1>::from_floats([1.0f32, 2.0], &device);
        let validation = Tensor::<1>::from_floats([1.0f32, 2.0, 3.0], &device);
        let calls = Cell::new(0);

        let error = OnnxExporter::new()
            .export_dynamic(
                &module,
                sample,
                validation,
                &[InputSpec::new([crate::export::AxisSpec::Static])],
                |module, input| {
                    calls.set(calls.get() + 1);
                    input + module.weight.val()
                },
            )
            .unwrap_err();

        assert_eq!(calls.get(), 0);
        assert!(matches!(error, ExportError::InvalidBoundary(_)));
    }
}
