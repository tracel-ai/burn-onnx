use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Artifact paths
// ---------------------------------------------------------------------------

/// Returns the artifacts directory for a model-check crate.
///
/// Used in `src/main.rs` (no cargo warnings).
pub fn artifacts_dir(model_name: &str) -> PathBuf {
    let base = match std::env::var("BURN_CACHE_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => dirs::cache_dir()
            .expect("could not determine cache directory")
            .join("burn-onnx"),
    };
    base.join("model-checks").join(model_name)
}

/// Returns the artifacts directory for a model-check crate, printing a
/// `cargo:warning` so the path is visible during builds.
///
/// Used in `build.rs`.
pub fn artifacts_dir_build(model_name: &str) -> PathBuf {
    let dir = artifacts_dir(model_name);
    println!(
        "cargo:warning=model-checks: artifacts dir = {}",
        dir.display()
    );
    dir
}

// ---------------------------------------------------------------------------
// Backend and device selection
// ---------------------------------------------------------------------------

/// Defines `MyBackend` type alias based on the active feature flag.
///
/// Expands to four `#[cfg(feature = "...")]` type aliases for wgpu,
/// flex, tch, and metal backends.
#[macro_export]
macro_rules! backend_type {
    () => {
        #[cfg(feature = "wgpu")]
        pub type MyBackend = burn::backend::Wgpu;

        #[cfg(feature = "flex")]
        pub type MyBackend = burn::backend::Flex;

        #[cfg(feature = "tch")]
        pub type MyBackend = burn::backend::LibTorch<f32>;

        #[cfg(feature = "metal")]
        pub type MyBackend = burn::backend::Metal;
    };
}

/// Returns the best available device for the active backend.
///
/// Override with `BURN_DEVICE=cpu|mps|cuda|cuda:N`.
///
/// Defaults:
/// - **wgpu / metal / flex**: `Default::default()` (already picks the best device)
/// - **tch**: MPS on macOS, Cuda(0) elsewhere
#[macro_export]
macro_rules! best_device {
    () => {{
        #[cfg(feature = "tch")]
        {
            use burn::backend::libtorch::LibTorchDevice;
            match std::env::var("BURN_DEVICE").ok().as_deref() {
                Some("cpu") => LibTorchDevice::Cpu,
                Some("mps") => LibTorchDevice::Mps,
                Some(s) if s.starts_with("cuda") => {
                    let idx = match s.strip_prefix("cuda:") {
                        Some(i) => match i.parse() {
                            Ok(n) => n,
                            Err(_) => {
                                eprintln!(
                                    "Warning: invalid CUDA index '{i}', defaulting to cuda:0"
                                );
                                0
                            }
                        },
                        None => 0,
                    };
                    LibTorchDevice::Cuda(idx)
                }
                _ => {
                    #[cfg(target_os = "macos")]
                    {
                        LibTorchDevice::Mps
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        LibTorchDevice::Cuda(0)
                    }
                }
            }
        }

        #[cfg(not(feature = "tch"))]
        {
            <MyBackend as burn::prelude::Backend>::Device::default()
        }
    }};
}
