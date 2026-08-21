#![cfg(feature = "export")]

use std::cell::RefCell;

use burn::module::{Module, Param};
use burn::nn::{
    Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
};
use burn::tensor::{Device, Int, Tensor, TensorData, Tolerance};
use burn::tensor::{
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions, PadMode},
};
use burn_onnx::export::{AxisSpec, ExportError, InputSpec, OnnxExporter};
use onnx_ir::ModelProto;
use ort::{session::Session, value::Tensor as OrtTensor};
use protobuf::Message;

mod models;
use models::resnet::ResNet18;

const RTOL: f32 = 1.0e-4;
const ATOL: f32 = 1.0e-5;

#[derive(Module, Debug)]
struct AddModule {
    weight: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct ReturnWeight {
    weight: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct AddTensorModule {
    offset: Tensor<1>,
}

impl AddTensorModule {
    fn forward(&self, input: Tensor<1>) -> Tensor<1> {
        input + self.offset.clone()
    }
}

impl AddModule {
    fn forward(&self, input: Tensor<1>) -> Tensor<1> {
        input + self.weight.val()
    }
}

impl ReturnWeight {
    fn forward(&self, input: Tensor<1>) -> (Tensor<1>, Tensor<1>) {
        (input + self.weight.val(), self.weight.val())
    }
}

#[derive(Module, Debug)]
struct Mlp {
    first: Linear,
    activation: Relu,
    second: Linear,
}

#[derive(Module, Debug)]
struct Flatten;

#[derive(Module, Debug)]
struct Identity;

impl Identity {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        input
    }
}

impl Flatten {
    fn forward(&self, input: Tensor<3>) -> Tensor<2> {
        let [batch, channels, width] = input.dims();
        input.reshape([batch, channels * width])
    }
}

#[derive(Module, Debug)]
struct SmallCnn {
    conv: Conv2d,
    activation: Relu,
    pool: MaxPool2d,
}

#[derive(Module, Debug)]
struct PoolAndReshape {
    pool: MaxPool2d,
}

impl PoolAndReshape {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        let pooled = self.pool.forward(input);
        let [_, _, height, width] = pooled.dims();
        pooled.reshape([height, width])
    }
}

#[derive(Module, Debug)]
struct PoolAndFull {
    pool: MaxPool2d,
}

impl PoolAndFull {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        let pooled = self.pool.forward(input);
        let [_, _, height, width] = pooled.dims();
        Tensor::full([height, width], 1.0, &pooled.device())
    }
}

#[derive(Module, Debug)]
struct BilinearResize;

impl BilinearResize {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        interpolate(
            input,
            [5, 7],
            InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
        )
    }
}

#[derive(Module, Debug)]
struct NearestResize;

impl NearestResize {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        interpolate(
            input,
            [5, 7],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    }
}

#[derive(Module, Debug)]
struct NearestExactDownsample;

impl NearestExactDownsample {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        interpolate(
            input,
            [2, 2],
            InterpolateOptions::new(InterpolateMode::NearestExact),
        )
    }
}

#[derive(Module, Debug)]
struct AddFull;

impl AddFull {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let full = Tensor::full([2, 3], 2.5, &input.device());
        input + full
    }
}

#[derive(Module, Debug)]
struct AddDynamicFull;

impl AddDynamicFull {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let [batch, width] = input.dims();
        let full = Tensor::full([batch, width], 2.5, &input.device());
        input + full
    }
}

#[derive(Module, Debug)]
struct ConstantPad;

impl ConstantPad {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        input.pad([(0, 0), (0, 0), (1, 2), (3, 1)], PadMode::Constant(2.5))
    }
}

#[derive(Module, Debug)]
struct PadFromOtherInput;

impl PadFromOtherInput {
    fn forward(&self, inputs: (Tensor<4>, Tensor<2>)) -> Tensor<4> {
        let [_, right] = inputs.1.dims();
        inputs
            .0
            .pad([(0, 0), (0, 0), (0, 0), (0, right)], PadMode::Constant(0.0))
    }
}

#[derive(Module, Debug)]
struct CatChannels;

impl CatChannels {
    fn forward(&self, inputs: (Tensor<4>, Tensor<4>, Tensor<4>)) -> Tensor<4> {
        Tensor::cat(vec![inputs.0, inputs.1, inputs.2], 1)
    }
}

#[derive(Module, Debug)]
struct CatWidths;

impl CatWidths {
    fn forward(&self, inputs: (Tensor<2>, Tensor<2>)) -> Tensor<2> {
        Tensor::cat(vec![inputs.0, inputs.1], 1)
    }
}

#[derive(Module, Debug)]
struct Neg;

impl Neg {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        -input
    }
}

#[derive(Module, Debug)]
struct NegSpatial;

impl NegSpatial {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        -input
    }
}

#[derive(Module, Debug)]
struct MixedDeviceOutput;

impl MixedDeviceOutput {
    fn forward(&self, input: Tensor<1>) -> (Tensor<1>, Tensor<1>) {
        (input, Tensor::zeros([2], &Device::default()))
    }
}

impl SmallCnn {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.pool
            .forward(self.activation.forward(self.conv.forward(input)))
    }
}

impl Mlp {
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.second
            .forward(self.activation.forward(self.first.forward(input)))
    }
}

fn run_ort(model: &[u8], shape: impl Into<Vec<i64>>, input: Vec<f32>) -> TensorData {
    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(model)
        .unwrap();
    let input = OrtTensor::from_array((shape.into(), input)).unwrap();
    let outputs = session.run(ort::inputs![input]).unwrap();
    let (shape, values) = outputs[0].try_extract_tensor::<f32>().unwrap();
    TensorData::new(
        values.to_vec(),
        shape
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn add_matches_burn() {
    let device = Device::default();
    let module = AddModule {
        weight: Param::from_data([2.0f32, 3.0], &device),
    };
    let input_values = vec![5.0f32, 7.0];
    let input = Tensor::<1>::from_floats(input_values.as_slice(), &device);
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, AddModule::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [2], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn parameter_output_matches_burn() {
    let device = Device::default();
    let module = ReturnWeight {
        weight: Param::from_data([2.0f32, 3.0], &device),
    };
    let input_values = vec![5.0f32, 7.0];
    let input = Tensor::<1>::from_floats(input_values.as_slice(), &device);
    let (expected_sum, expected_weight) = module.forward(input.clone());
    let model = OnnxExporter::new()
        .export(&module, input, ReturnWeight::forward)
        .unwrap();

    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(model.as_bytes())
        .unwrap();
    let outputs = session
        .run(ort::inputs![
            OrtTensor::from_array(([2], input_values)).unwrap()
        ])
        .unwrap();
    let (_, sum) = outputs[0].try_extract_tensor::<f32>().unwrap();
    let (_, weight) = outputs[1].try_extract_tensor::<f32>().unwrap();
    TensorData::new(sum.to_vec(), [2])
        .assert_approx_eq::<f32>(&expected_sum.into_data(), Tolerance::default());
    TensorData::new(weight.to_vec(), [2])
        .assert_approx_eq::<f32>(&expected_weight.into_data(), Tolerance::default());
}

#[test]
fn mixed_device_output_returns_error() {
    let device = Device::default();
    let input = Tensor::<1>::zeros([2], &device);

    assert!(matches!(
        OnnxExporter::new().export(&MixedDeviceOutput, input, MixedDeviceOutput::forward),
        Err(ExportError::InvalidBoundary(reason))
            if reason.contains("capture device")
    ));
}

#[test]
fn pass_through_matches_burn() {
    let device = Device::default();
    let input = Tensor::<1, Int>::arange(0..6, &device)
        .float()
        .reshape([2, 3]);
    let input_values = input.clone().into_data().try_to_vec::<f32>().unwrap();
    let expected = Identity.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&Identity, input, Identity::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [2, 3], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn small_mlp_matches_burn() {
    let device = Device::default();
    let module = Mlp {
        first: LinearConfig::new(4, 3).init(&device),
        activation: Relu::new(),
        second: LinearConfig::new(3, 2).init(&device),
    };
    let input_values = vec![1.0f32, 2.0, 3.0, 4.0, -2.0, 0.5, 1.5, 3.0];
    let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, 3.0]], &device);
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, Mlp::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [2, 4], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn dynamic_reshape_matches_burn_at_runtime_shape() {
    let device = Device::default();
    let module = Flatten;
    let sample = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([2, 3, 4]);
    let validation = Tensor::<1, Int>::arange(0..60, &device)
        .float()
        .reshape([5, 3, 4]);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch"),
        AxisSpec::Static,
        AxisSpec::Static,
    ])];

    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, Flatten::forward)
        .unwrap();

    let runtime_input = Tensor::<1, Int>::arange(0..84, &device)
        .float()
        .reshape([7, 3, 4]);
    let runtime_values = runtime_input
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let expected = module.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [7, 3, 4], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn dynamic_export_isolates_capture_devices() {
    let device = Device::default();
    let module = AddTensorModule {
        offset: Tensor::from_floats([2.0], &device),
    };
    let sample = Tensor::<1>::from_floats([1.0, 2.0], &device);
    let validation = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
    let capture_devices = RefCell::new(Vec::new());

    let model = OnnxExporter::new()
        .export_dynamic(
            &module,
            sample,
            validation,
            &[InputSpec::new([AxisSpec::dynamic("batch_size")])],
            |module, input| {
                capture_devices.borrow_mut().push(input.device());
                module.forward(input)
            },
        )
        .unwrap();

    let capture_devices = capture_devices.into_inner();
    assert_eq!(capture_devices.len(), 2);
    assert_ne!(capture_devices[0], capture_devices[1]);

    let runtime_values = vec![4.0, 5.0, 6.0, 7.0];
    let runtime_input = Tensor::<1>::from_floats(runtime_values.as_slice(), &device);
    let expected = module.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [4], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn small_cnn_matches_burn() {
    let device = Device::default();
    let module = SmallCnn {
        conv: Conv2dConfig::new([1, 2], [3, 3]).init(&device),
        activation: Relu::new(),
        pool: MaxPool2dConfig::new([2, 2]).init(),
    };
    let input = (Tensor::<1, Int>::arange(0..25, &device).float() / 10.0).reshape([1, 1, 5, 5]);
    let input_values = input.clone().into_data().try_to_vec::<f32>().unwrap();
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, SmallCnn::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [1, 1, 5, 5], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn interpolate_matches_burn() {
    let device = Device::default();
    let input = Tensor::<1, Int>::arange(0..12, &device)
        .float()
        .reshape([1, 1, 3, 4]);
    let input_values = input.clone().into_data().try_to_vec::<f32>().unwrap();
    let expected = BilinearResize.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&BilinearResize, input, BilinearResize::forward)
        .unwrap();
    let actual = run_ort(model.as_bytes(), [1, 1, 3, 4], input_values.clone());
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));

    let input = Tensor::<1, Int>::arange(0..12, &device)
        .float()
        .reshape([1, 1, 3, 4]);
    let expected = NearestResize.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&NearestResize, input, NearestResize::forward)
        .unwrap();
    let actual = run_ort(model.as_bytes(), [1, 1, 3, 4], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn nearest_exact_downsample_uses_burn_rounding() {
    let device = Device::default();
    let input = Tensor::<1, Int>::arange(0..16, &device)
        .float()
        .reshape([1, 1, 4, 4]);
    let input_values = input.clone().into_data().try_to_vec::<f32>().unwrap();
    // Burn's nearest-exact rule selects source indices 1 and 3 on each axis.
    let expected = TensorData::new(vec![5.0f32, 7.0, 13.0, 15.0], [1, 1, 2, 2]);
    let model = OnnxExporter::new()
        .export(
            &NearestExactDownsample,
            input,
            NearestExactDownsample::forward,
        )
        .unwrap();

    let actual = run_ort(model.as_bytes(), [1, 1, 4, 4], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn full_matches_burn() {
    let device = Device::default();
    let input_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::<2>::from_data(TensorData::new(input_values.clone(), [2, 3]), &device);
    let expected = AddFull.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&AddFull, input, AddFull::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [2, 3], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn dynamic_full_matches_burn_at_runtime_shape() {
    let device = Device::default();
    let sample = Tensor::<1, Int>::arange(0..6, &device)
        .float()
        .reshape([2, 3]);
    let validation = Tensor::<1, Int>::arange(0..15, &device)
        .float()
        .reshape([5, 3]);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch_size"),
        AxisSpec::Static,
    ])];

    let model = OnnxExporter::new()
        .export_dynamic(
            &AddDynamicFull,
            sample,
            validation,
            &specs,
            AddDynamicFull::forward,
        )
        .unwrap();

    let runtime_input = Tensor::<1, Int>::arange(0..21, &device)
        .float()
        .reshape([7, 3]);
    let runtime_values = runtime_input
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let expected = AddDynamicFull.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [7, 3], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn constant_pad_matches_burn() {
    let device = Device::default();
    let input_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let input =
        Tensor::<4>::from_data(TensorData::new(input_values.clone(), [1, 1, 2, 3]), &device);
    let expected = ConstantPad.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&ConstantPad, input, ConstantPad::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [1, 1, 2, 3], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn dynamic_constant_pad_matches_burn_at_runtime_shape() {
    let device = Device::default();
    let sample = Tensor::<4>::zeros([1, 1, 2, 3], &device);
    let validation = Tensor::<4>::zeros([2, 1, 4, 5], &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch_size"),
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::dynamic("width"),
    ])];

    let model = OnnxExporter::new()
        .export_dynamic(
            &ConstantPad,
            sample,
            validation,
            &specs,
            ConstantPad::forward,
        )
        .unwrap();

    let runtime_input =
        (Tensor::<1, Int>::arange(0..3 * 6 * 7, &device).float() / 10.0).reshape([3, 1, 6, 7]);
    let runtime_values = runtime_input
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let expected = ConstantPad.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [3, 1, 6, 7], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn dynamic_pad_from_other_input_is_rejected() {
    let device = Device::default();
    let sample = (
        Tensor::<4>::zeros([1, 1, 2, 3], &device),
        Tensor::<2>::zeros([1, 1], &device),
    );
    let validation = (
        Tensor::<4>::zeros([1, 1, 2, 3], &device),
        Tensor::<2>::zeros([1, 2], &device),
    );
    let specs = [
        InputSpec::new([
            AxisSpec::Static,
            AxisSpec::Static,
            AxisSpec::Static,
            AxisSpec::Static,
        ]),
        InputSpec::new([AxisSpec::Static, AxisSpec::dynamic("right_pad")]),
    ];

    assert!(matches!(
        OnnxExporter::new().export_dynamic(
            &PadFromOtherInput,
            sample,
            validation,
            &specs,
            PadFromOtherInput::forward,
        ),
        Err(ExportError::DynamicShapeLost { .. })
    ));
}

#[test]
fn dynamic_output_preserves_distinct_symbols_with_equal_capture_sizes() {
    let device = Device::default();
    let sample = Tensor::<4>::zeros([1, 1, 2, 2], &device);
    let validation = Tensor::<4>::zeros([2, 1, 4, 4], &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch_size"),
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::dynamic("width"),
    ])];
    let model = OnnxExporter::new()
        .export_dynamic(&NegSpatial, sample, validation, &specs, NegSpatial::forward)
        .unwrap();

    let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
    let output_shape = &model.graph.output[0]
        .type_
        .as_ref()
        .unwrap()
        .tensor_type()
        .shape
        .dim;
    assert_eq!(output_shape[0].dim_param(), "batch_size");
    assert_eq!(output_shape[2].dim_param(), "height");
    assert_eq!(output_shape[3].dim_param(), "width");
}

#[test]
fn cat_matches_burn() {
    let device = Device::default();
    let first_values = vec![1.0f32, 2.0, 3.0, 4.0];
    let second_values = vec![5.0f32, 6.0, 7.0, 8.0];
    let third_values = vec![9.0f32, 10.0, 11.0, 12.0];
    let inputs = (
        Tensor::<4>::from_data(TensorData::new(first_values.clone(), [1, 1, 2, 2]), &device),
        Tensor::<4>::from_data(
            TensorData::new(second_values.clone(), [1, 1, 2, 2]),
            &device,
        ),
        Tensor::<4>::from_data(TensorData::new(third_values.clone(), [1, 1, 2, 2]), &device),
    );
    let expected = CatChannels.forward(inputs.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&CatChannels, inputs, CatChannels::forward)
        .unwrap();

    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(model.as_bytes())
        .unwrap();
    let outputs = session
        .run(ort::inputs![
            OrtTensor::from_array(([1, 1, 2, 2], first_values)).unwrap(),
            OrtTensor::from_array(([1, 1, 2, 2], second_values)).unwrap(),
            OrtTensor::from_array(([1, 1, 2, 2], third_values)).unwrap(),
        ])
        .unwrap();
    let (shape, values) = outputs[0].try_extract_tensor::<f32>().unwrap();
    let actual = TensorData::new(
        values.to_vec(),
        shape
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>(),
    );
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn neg_matches_burn() {
    let device = Device::default();
    let input_values = vec![-3.0f32, -0.5, 0.0, 2.0, 4.5, 10.0];
    let input = Tensor::<2>::from_data(TensorData::new(input_values.clone(), [2, 3]), &device);
    let expected = Neg.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&Neg, input, Neg::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [2, 3], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
}

#[test]
fn dynamic_small_cnn_matches_burn_at_runtime_shape() {
    let device = Device::default();
    let module = SmallCnn {
        conv: Conv2dConfig::new([1, 2], [3, 3]).init(&device),
        activation: Relu::new(),
        pool: MaxPool2dConfig::new([2, 2]).init(),
    };
    let sample = Tensor::<4>::zeros([1, 1, 5, 5], &device);
    let validation = Tensor::<4>::zeros([2, 1, 7, 7], &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch"),
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::dynamic("width"),
    ])];
    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, SmallCnn::forward)
        .unwrap();

    let runtime_input =
        (Tensor::<1, Int>::arange(0..243, &device).float() / 100.0).reshape([3, 1, 9, 9]);
    let runtime_values = runtime_input
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let expected = module.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [3, 1, 9, 9], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(RTOL, ATOL));
}

#[test]
fn dynamic_pool_reshape_handles_colliding_capture_shapes() {
    let device = Device::default();
    let module = PoolAndReshape {
        pool: MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init(),
    };
    // Heights 4 and 5 both pool to 2, but the runtime height 6 pools to 3.
    let sample = Tensor::<4>::zeros([1, 1, 4, 4], &device);
    let validation = Tensor::<4>::zeros([1, 1, 5, 4], &device);
    let specs = [InputSpec::new([
        AxisSpec::Static,
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::Static,
    ])];
    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, PoolAndReshape::forward)
        .unwrap();

    let runtime_input = Tensor::<1, Int>::arange(0..24, &device)
        .float()
        .reshape([1, 1, 6, 4]);
    let runtime_values = runtime_input
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let expected = module.forward(runtime_input).into_data();
    let actual = run_ort(model.as_bytes(), [1, 1, 6, 4], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());

    let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
    let output_shape = &model.graph.output[0]
        .type_
        .as_ref()
        .unwrap()
        .tensor_type()
        .shape
        .dim;
    assert!(output_shape[0].has_dim_param());
}

#[test]
fn dynamic_cat_handles_colliding_capture_shapes() {
    let device = Device::default();
    // The two dynamic widths vary inversely, so both captures concatenate to width 7.
    let sample = (
        Tensor::<2>::zeros([1, 2], &device),
        Tensor::<2>::zeros([1, 5], &device),
    );
    let validation = (
        Tensor::<2>::zeros([1, 3], &device),
        Tensor::<2>::zeros([1, 4], &device),
    );
    let specs = [
        InputSpec::new([AxisSpec::Static, AxisSpec::dynamic("first_width")]),
        InputSpec::new([AxisSpec::Static, AxisSpec::dynamic("second_width")]),
    ];
    let model = OnnxExporter::new()
        .export_dynamic(&CatWidths, sample, validation, &specs, CatWidths::forward)
        .unwrap();

    let first = Tensor::<1, Int>::arange(0..4, &device)
        .float()
        .reshape([1, 4]);
    let second = (Tensor::<1, Int>::arange(0..6, &device).float() + 10.0).reshape([1, 6]);
    let first_values = first.clone().into_data().try_to_vec::<f32>().unwrap();
    let second_values = second.clone().into_data().try_to_vec::<f32>().unwrap();
    let expected = CatWidths.forward((first, second)).into_data();

    let mut session = Session::builder()
        .unwrap()
        .commit_from_memory(model.as_bytes())
        .unwrap();
    let outputs = session
        .run(ort::inputs![
            OrtTensor::from_array(([1, 4], first_values)).unwrap(),
            OrtTensor::from_array(([1, 6], second_values)).unwrap(),
        ])
        .unwrap();
    let (shape, values) = outputs[0].try_extract_tensor::<f32>().unwrap();
    let actual = TensorData::new(
        values.to_vec(),
        shape
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>(),
    );
    actual.assert_approx_eq::<f32>(&expected, Tolerance::default());

    let model = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
    let output_shape = &model.graph.output[0]
        .type_
        .as_ref()
        .unwrap()
        .tensor_type()
        .shape
        .dim;
    assert_eq!(output_shape[1].dim_param(), "output_0_dim_1");
}

#[test]
fn dynamic_full_rejects_colliding_capture_shapes() {
    let device = Device::default();
    let module = PoolAndFull {
        pool: MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init(),
    };
    let sample = Tensor::<4>::zeros([1, 1, 4, 4], &device);
    let validation = Tensor::<4>::zeros([1, 1, 5, 4], &device);
    let specs = [InputSpec::new([
        AxisSpec::Static,
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::Static,
    ])];

    assert!(matches!(
        OnnxExporter::new().export_dynamic(
            &module,
            sample,
            validation,
            &specs,
            PoolAndFull::forward,
        ),
        Err(ExportError::DynamicShapeLost { axis: 0, .. })
    ));
}

#[test]
fn resnet18_matches_burn() {
    let device = Device::default();
    let module = ResNet18::new(10, &device);
    let input_values = (0..3 * 64 * 64)
        .map(|value| (value % 251) as f32 / 251.0)
        .collect::<Vec<_>>();
    let input = Tensor::<4>::from_data(
        TensorData::new(input_values.clone(), [1, 3, 64, 64]),
        &device,
    );
    let expected = module.forward(input.clone()).into_data();
    let model = OnnxExporter::new()
        .export(&module, input, ResNet18::forward)
        .unwrap();

    let model_proto = ModelProto::parse_from_bytes(model.as_bytes()).unwrap();
    let batch_norm_count = model_proto
        .graph
        .node
        .iter()
        .filter(|node| node.op_type == "BatchNormalization")
        .count();
    assert_eq!(batch_norm_count, 20);

    let actual = run_ort(model.as_bytes(), [1, 3, 64, 64], input_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1.0e-3, 1.0e-5));
}

#[test]
fn dynamic_resnet18_matches_burn_at_runtime_shape() {
    let device = Device::default();
    let module = ResNet18::new(10, &device);
    let sample = Tensor::<4>::zeros([1, 3, 32, 32], &device);
    let validation = Tensor::<4>::zeros([2, 3, 40, 48], &device);
    let specs = [InputSpec::new([
        AxisSpec::dynamic("batch_size"),
        AxisSpec::Static,
        AxisSpec::dynamic("height"),
        AxisSpec::dynamic("width"),
    ])];
    let runtime_values = (0..3 * 3 * 48 * 56)
        .map(|value| (value % 251) as f32 / 251.0)
        .collect::<Vec<_>>();
    let runtime_input = Tensor::<4>::from_data(
        TensorData::new(runtime_values.clone(), [3, 3, 48, 56]),
        &device,
    );
    let expected = module.forward(runtime_input).into_data();

    let model = OnnxExporter::new()
        .export_dynamic(&module, sample, validation, &specs, ResNet18::forward)
        .unwrap();

    let actual = run_ort(model.as_bytes(), [3, 3, 48, 56], runtime_values);
    actual.assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(1.0e-3, 1.0e-5));
}
