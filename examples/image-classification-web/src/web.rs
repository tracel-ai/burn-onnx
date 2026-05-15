#![allow(clippy::new_without_default)]

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::convert::Into;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::model::{label::LABELS, normalizer::Normalizer, squeezenet::Model as SqueezenetModel};

use burn::{
    prelude::*,
    tensor::{FlexDevice, WgpuDevice, activation::softmax},
};

use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_time::Instant;

// Global value to ensure that the wgpu backend is initialized at most once
static WGPU_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[wasm_bindgen(start)]
pub fn start() {
    // Initialize the logger so that the logs are printed to the console
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

/// The image is 224x224 pixels with 3 channels (RGB)
const HEIGHT: usize = 224;
const WIDTH: usize = 224;
const CHANNELS: usize = 3;

/// The image classifier
#[wasm_bindgen]
pub struct ImageClassifier {
    model: Model,
}

#[wasm_bindgen]
impl ImageClassifier {
    /// Constructor called by JavaScripts with the new keyword.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Initializing the image classifier");
        let device: Device = FlexDevice.into();
        Self {
            model: Model::new(&device),
        }
    }

    /// Runs inference on the image
    pub async fn inference(&self, input: &[f32]) -> Result<JsValue, JsValue> {
        log::info!("Running inference on the image");

        let start = Instant::now();
        let result = self.model.forward(input).await;
        let duration = start.elapsed();

        log::debug!("Inference is completed in {duration:?}");

        top_5_classes(result)
    }

    /// Sets the backend to Flex (pure-Rust CPU backend)
    pub async fn set_backend_flex(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Flex backend");
        let start = Instant::now();
        let device: Device = FlexDevice.into();
        self.model = Model::new(&device);
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Flex backend in {duration:?}");
        Ok(())
    }

    /// Sets the backend to Wgpu
    pub async fn set_backend_wgpu(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();
        let wgpu_device = WgpuDevice::default();

        // First-time wgpu device init is handled internally by the new burn dispatch.
        // The compare_exchange guard keeps the original "init once" contract in case
        // we need to add an explicit setup hook later.
        let _ = WGPU_INITIALIZED.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst);
        let device: Device = wgpu_device.into();
        self.model = Model::new(&device);
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Wgpu backend in {duration:?}");

        log::debug!("Warming up the model");
        let start = Instant::now();
        let _ = self.inference(&[0.0; HEIGHT * WIDTH * CHANNELS]).await;
        let duration = start.elapsed();
        log::debug!("Warming up is completed in {duration:?}");
        Ok(())
    }
}

/// The image classifier model
pub struct Model {
    model: SqueezenetModel,
    normalizer: Normalizer,
    device: Device,
}

impl Model {
    /// Constructor
    pub fn new(device: &Device) -> Self {
        Self {
            model: SqueezenetModel::from_embedded(device),
            normalizer: Normalizer::new(device),
            device: device.clone(),
        }
    }

    /// Normalizes input and runs inference on the image
    pub async fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Reshape from the 1D array to 3d tensor [ width, height, channels]
        let input =
            Tensor::<1>::from_floats(input, &self.device).reshape([1, CHANNELS, HEIGHT, WIDTH]);

        // Normalize input: make between [-1,1] and make the mean=0 and std=1
        let input = self.normalizer.normalize(input);

        // Run the tensor input through the model
        let output = self.model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let probabilities = softmax(output, 1);

        // Forces the result to be computed
        probabilities
            .into_data_async()
            .await
            .unwrap()
            .convert::<f32>()
            .to_vec()
            .unwrap()
    }
}

#[wasm_bindgen]
#[derive(Serialize)]
pub struct InferenceResult {
    index: usize,
    probability: f32,
    label: String,
}

/// Returns the top 5 classes and convert them into a JsValue
fn top_5_classes(probabilities: Vec<f32>) -> Result<JsValue, JsValue> {
    // Convert the probabilities into a vector of (index, probability)
    let mut probabilities: Vec<_> = probabilities.iter().enumerate().collect();

    // Sort the probabilities in descending order
    probabilities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // Take the top 5 probabilities
    probabilities.truncate(5);

    // Convert the probabilities into InferenceResult
    let result: Vec<InferenceResult> = probabilities
        .into_iter()
        .map(|(index, probability)| InferenceResult {
            index,
            probability: *probability,
            label: LABELS[index].to_string(),
        })
        .collect();

    // Convert the InferenceResult into a JsValue
    Ok(serde_wasm_bindgen::to_value(&result)?)
}
