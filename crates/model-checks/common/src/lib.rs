use std::path::PathBuf;

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

/// Defines `MyBackend` type alias based on the active feature flag.
///
/// Expands to four `#[cfg(feature = "...")]` type aliases for wgpu,
/// ndarray, tch, and metal backends.
#[macro_export]
macro_rules! backend_type {
    () => {
        #[cfg(feature = "wgpu")]
        pub type MyBackend = burn::backend::Wgpu;

        #[cfg(feature = "ndarray")]
        pub type MyBackend = burn::backend::NdArray<f32>;

        #[cfg(feature = "tch")]
        pub type MyBackend = burn::backend::LibTorch<f32>;

        #[cfg(feature = "metal")]
        pub type MyBackend = burn::backend::Metal;
    };
}
