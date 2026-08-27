//! Lower resolved Burn export graphs into ONNX protobuf models.
//!
//! Lowering is intentionally independent from trace-based shape inference. The
//! module receives a [`crate::export::ResolvedExportGraph`], validates its value
//! bindings, initializes graph boundaries and embedded values, and then
//! dispatches each Burn operation to an operation-family lowerer.
//!
//! [`context::LoweringContext`] owns protobuf construction and deterministic
//! naming. The family modules only translate operations.

mod base;
mod context;
mod direct;
mod module;
mod numeric;

use std::collections::BTreeMap;

use burn::backend::ir::{OperationIr, ScalarIr, TensorId, TensorIr};
use burn::backend::{DType, TensorData};
use context::LoweringContext;
use hashbrown::{HashMap, HashSet};
use onnx_ir::{GraphProto, TensorProto, TypeProto, ValueInfoProto};
use protobuf::MessageField;

use crate::export::{ExportError, OnnxModel, Opset, ResolvedExportGraph};

/// ONNX IR version emitted by this exporter.
const ONNX_IR_VERSION: i64 = 8;
/// Maximum protobuf payload supported by embedded ONNX tensor data.
const MAX_EMBEDDED_PROTOBUF_BYTES: u64 = i32::MAX as u64;

pub(crate) fn export_graph_with_bindings_and_opset(
    graph: &ResolvedExportGraph,
    values: &BTreeMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
    opset: Opset,
) -> Result<OnnxModel, ExportError> {
    validate_bindings(graph, values, runtime_inputs, initializer_names)?;
    let runtime_set: HashSet<_> = runtime_inputs.iter().copied().collect();
    let embedded_bytes = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .map(|(_, value)| value.bytes.len() as u64)
        .sum::<u64>();
    if embedded_bytes > MAX_EMBEDDED_PROTOBUF_BYTES {
        return Err(ExportError::Serialization(format!(
            "embedded tensor data is {embedded_bytes} bytes, exceeding the protobuf limit of {MAX_EMBEDDED_PROTOBUF_BYTES} bytes"
        )));
    }

    let mut proto = GraphProto::new();
    proto.name = "burn_graph".into();
    for &id in &graph.graph.inputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        proto
            .input
            .push(value_info(tensor, &graph.dynamic_axes, initializer_names)?);
    }
    for &id in &graph.graph.outputs {
        let tensor = find_tensor(graph, id).ok_or(ExportError::MissingValue(id))?;
        let mut output = value_info(tensor, &graph.dynamic_axes, initializer_names)?;
        if graph.graph.inputs.contains(&id) {
            output.name = pass_through_output_name(id);
        }
        proto.output.push(output);
    }
    let mut initializers: Vec<_> = values
        .iter()
        .filter(|(id, _)| !runtime_set.contains(*id))
        .collect();
    initializers.sort_by(|(lhs, _), (rhs, _)| {
        tensor_name(**lhs, initializer_names).cmp(&tensor_name(**rhs, initializer_names))
    });
    for (&id, data) in initializers {
        let mut initializer = TensorProto::new();
        initializer.name = tensor_name(id, initializer_names);
        initializer.data_type = onnx_dtype_parts(id, data.dtype)?;
        initializer.dims = data.shape.iter().map(|dim| *dim as i64).collect();
        initializer.raw_data = bytes::Bytes::copy_from_slice(data.bytes.as_ref());
        proto.initializer.push(initializer);
    }

    let mut context = LoweringContext::new(graph, proto, initializer_names, opset);
    for (position, &id) in graph.graph.outputs.iter().enumerate() {
        if graph.graph.inputs.contains(&id) {
            let input = context.tensor_name(id);
            context.node(
                format!("output_{position}_identity"),
                "Identity",
                vec![input],
                vec![pass_through_output_name(id)],
            );
        }
    }
    for (index, operation) in graph.graph.operations.iter().enumerate() {
        lower_operation(&mut context, index, operation)?;
    }
    context.finish()
}

fn lower_operation(
    context: &mut LoweringContext<'_>,
    index: usize,
    operation: &OperationIr,
) -> Result<(), ExportError> {
    // Init records carry tensor metadata and initialized values, but do not represent an ONNX
    // computation. Keeping them in the graph also preserves metadata for pass-through boundaries.
    if matches!(operation, OperationIr::Init(_)) {
        return Ok(());
    }
    if numeric::lower(context, index, operation)?
        || base::lower(context, index, operation)?
        || module::lower(context, index, operation)?
        || direct::lower(context, index, operation)?
    {
        return Ok(());
    }
    Err(ExportError::UnsupportedOperation {
        operation: index,
        kind: format!("{operation:?}"),
    })
}

fn validate_bindings(
    graph: &ResolvedExportGraph,
    values: &BTreeMap<TensorId, TensorData>,
    runtime_inputs: &[TensorId],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<(), ExportError> {
    let mut runtime = HashSet::new();
    for &id in runtime_inputs {
        if !runtime.insert(id) {
            return Err(ExportError::InvalidBoundary(format!(
                "duplicate runtime input tensor {id}"
            )));
        }
        if !graph.graph.inputs.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} is not a declared graph input"
            )));
        }
    }
    if runtime.len() != graph.graph.inputs.len() {
        return Err(ExportError::InvalidBoundary(
            "every declared graph input must have one runtime input binding".into(),
        ));
    }
    for (&id, name) in initializer_names {
        if runtime.contains(&id) {
            return Err(ExportError::InvalidBoundary(format!(
                "runtime input tensor {id} cannot also have an initializer name"
            )));
        }
        if !values.contains_key(&id) {
            return Err(ExportError::MissingValue(id));
        }
        if name.is_empty() {
            return Err(ExportError::InvalidBoundary(format!(
                "initializer tensor {id} has an empty name"
            )));
        }
    }

    let mut names = HashMap::<String, TensorId>::new();
    let ids = graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .map(|tensor| tensor.id)
        .chain(values.keys().copied());
    for id in ids {
        let name = tensor_name(id, initializer_names);
        if let Some(previous) = names.insert(name.clone(), id)
            && previous != id
        {
            return Err(ExportError::InvalidBoundary(format!(
                "ONNX value name `{name}` is shared by tensors {previous} and {id}"
            )));
        }
    }

    for (&id, data) in values {
        let Some(tensor) = find_tensor(graph, id) else {
            continue;
        };
        if tensor.dtype != data.dtype || tensor.shape != data.shape {
            return Err(ExportError::InvalidValue {
                tensor: id,
                reason: format!(
                    "graph metadata is {:?} {:?}, initialized value is {:?} {:?}",
                    tensor.dtype, tensor.shape, data.dtype, data.shape
                ),
            });
        }
    }
    Ok(())
}

fn value_info(
    tensor: &TensorIr,
    dynamic_axes: &[crate::export::DynamicAxis],
    initializer_names: &HashMap<TensorId, String>,
) -> Result<ValueInfoProto, ExportError> {
    let mut info = ValueInfoProto::new();
    info.name = tensor_name(tensor.id, initializer_names);
    let mut ty = TypeProto::new();
    let tensor_type = ty.mut_tensor_type();
    tensor_type.elem_type = onnx_dtype_parts(tensor.id, tensor.dtype)?;
    let shape = tensor_type.shape.mut_or_insert_default();
    for (axis, &dim) in tensor.shape.iter().enumerate() {
        shape.dim.push(Default::default());
        let dimension = shape.dim.last_mut().unwrap();
        if let Some(dynamic) = dynamic_axes
            .iter()
            .find(|dynamic| dynamic.tensor == tensor.id && dynamic.axis == axis)
        {
            dimension.set_dim_param(dynamic.symbol.clone());
        } else {
            dimension.set_dim_value(dim as i64);
        }
    }
    info.type_ = MessageField::some(ty);
    Ok(info)
}

fn onnx_dtype_parts(tensor: TensorId, dtype: DType) -> Result<i32, ExportError> {
    // TensorProto.DataType numeric values from the ONNX specification.
    match dtype {
        DType::F32 => Ok(1),
        DType::U8 => Ok(2),
        DType::I8 => Ok(3),
        DType::I16 => Ok(5),
        DType::I32 => Ok(6),
        DType::I64 => Ok(7),
        DType::Bool(_) => Ok(9),
        DType::F16 => Ok(10),
        DType::F64 => Ok(11),
        DType::U32 => Ok(12),
        DType::U64 => Ok(13),
        DType::BF16 => Ok(16),
        dtype => Err(ExportError::UnsupportedDType {
            tensor,
            dtype: format!("{dtype:?}"),
        }),
    }
}

fn find_tensor(graph: &ResolvedExportGraph, id: TensorId) -> Option<&TensorIr> {
    graph
        .graph
        .operations
        .iter()
        .flat_map(OperationIr::nodes)
        .find(|tensor| tensor.id == id)
}

fn name(id: TensorId) -> String {
    format!("tensor_{}", id.value())
}

fn pass_through_output_name(id: TensorId) -> String {
    format!("{}_output", name(id))
}

fn tensor_name(id: TensorId, initializer_names: &HashMap<TensorId, String>) -> String {
    initializer_names
        .get(&id)
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| name(id))
}

fn scalar_tensor(
    dtype: DType,
    value: ScalarIr,
    tensor: TensorId,
) -> Result<TensorProto, ExportError> {
    let mut initializer = TensorProto::new();
    initializer.data_type = onnx_dtype_parts(tensor, dtype)?;
    initializer.dims = vec![1];
    let bytes = match dtype {
        DType::F32 => value.elem::<f32>().to_le_bytes().to_vec(),
        DType::F64 => value.elem::<f64>().to_le_bytes().to_vec(),
        DType::I32 => value.elem::<i32>().to_le_bytes().to_vec(),
        DType::I64 => value.elem::<i64>().to_le_bytes().to_vec(),
        dtype => {
            return Err(ExportError::UnsupportedDType {
                tensor,
                dtype: format!("{dtype:?} scalar initializer"),
            });
        }
    };
    initializer.raw_data = bytes::Bytes::from(bytes);
    Ok(initializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Shape;
    use burn::backend::ir::{BinaryOpIr, GraphIr, NumericOperationIr};
    use onnx_ir::ModelProto;
    use protobuf::Message;

    fn tensor(id: u64) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), Shape::new([2, 3]), DType::F32)
    }

    #[test]
    fn writes_parseable_opset_18_model() {
        let graph = GraphIr::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor(1),
                rhs: tensor(2),
                out: tensor(3),
            }),
        )]);
        let graph = ResolvedExportGraph {
            graph,
            shapes: vec![],
            dynamic_axes: vec![],
        };
        let runtime_inputs = graph.graph.inputs.clone();
        let model = export_graph_with_bindings_and_opset(
            &graph,
            &BTreeMap::new(),
            &runtime_inputs,
            &HashMap::new(),
            Opset::default(),
        )
        .unwrap();
        let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
        assert_eq!(model.ir_version, ONNX_IR_VERSION);
        assert_eq!(model.opset_import[0].version, Opset::default().version());
        assert_eq!(model.graph.node[0].op_type, "Add");
        assert_eq!(model.graph.input.len(), 2);
        assert_eq!(model.graph.output.len(), 1);
    }
}
