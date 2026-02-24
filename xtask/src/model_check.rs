use std::collections::HashMap;
use std::path::PathBuf;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct ModelCheckArgs {
    /// Model id to operate on, e.g. "qwen-2.5" (default: all non-blocked models).
    #[arg(long)]
    pub model: Option<String>,

    /// Cargo features to pass (default: ndarray).
    #[arg(long, default_value = "ndarray")]
    pub features: String,

    /// Build in debug mode instead of release.
    #[arg(long)]
    pub debug: bool,

    /// Stop at the first model failure instead of continuing.
    #[arg(long)]
    pub fail_fast: bool,

    #[command(subcommand)]
    pub command: Option<ModelCheckSubCommand>,
}

#[derive(clap::Subcommand)]
pub enum ModelCheckSubCommand {
    /// Download model artifacts (runs get_model.py).
    Download,
    /// Build the model-check crate.
    Build,
    /// Run the model-check binary.
    Run,
    /// Download, build, and run (default).
    All,
}

struct ModelInfo {
    /// Unique identifier used by `--model <id>`.
    id: &'static str,
    /// Directory under crates/model-checks/.
    dir: &'static str,
    name: &'static str,
    /// Optional (env_var, default_value) for models with selectable variants.
    env: Option<(&'static str, &'static str)>,
    /// Extra arguments passed to get_model.py during download.
    download_args: &'static [&'static str],
    /// Skipped in default "all" runs; still runnable with `--model <id>`.
    blocked: bool,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "silero-vad",
        dir: "silero-vad",
        name: "Silero VAD",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "all-minilm-l6-v2",
        dir: "all-minilm-l6-v2",
        name: "all-MiniLM-L6-v2",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "clip-vit-b-32-text",
        dir: "clip-vit-b-32-text",
        name: "CLIP ViT-B-32 text",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "clip-vit-b-32-vision",
        dir: "clip-vit-b-32-vision",
        name: "CLIP ViT-B-32 vision",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "modernbert-base",
        dir: "modernbert-base",
        name: "ModernBERT-base",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "rf-detr",
        dir: "rf-detr",
        name: "RF-DETR Small",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "depth-anything-v2",
        dir: "depth-anything-v2",
        name: "Depth-Anything-v2",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "albert",
        dir: "albert",
        name: "ALBERT",
        env: Some(("ALBERT_MODEL", "albert-base-v2")),
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "yolo",
        dir: "yolo",
        name: "YOLO",
        env: Some(("YOLO_MODEL", "yolov8n")),
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "mediapipe-face-detector",
        dir: "mediapipe-face-detector",
        name: "MediaPipe Face Detector",
        env: None,
        download_args: &[],
        blocked: false,
    },
    ModelInfo {
        id: "smollm",
        dir: "smollm",
        name: "SmolLM 135M",
        env: Some(("SMOLLM_MODEL", "smollm-135m")),
        download_args: &["--model", "smollm-135m"],
        blocked: false,
    },
    ModelInfo {
        id: "smollm2",
        dir: "smollm",
        name: "SmolLM2 135M",
        env: Some(("SMOLLM_MODEL", "smollm2-135m")),
        download_args: &["--model", "smollm2-135m"],
        blocked: false,
    },
    ModelInfo {
        id: "qwen-1.5",
        dir: "qwen",
        name: "Qwen1.5 0.5B",
        env: Some(("QWEN_MODEL", "qwen1.5-0.5b")),
        download_args: &["--model", "qwen1.5-0.5b"],
        blocked: true,
    },
    ModelInfo {
        id: "qwen-2.5",
        dir: "qwen",
        name: "Qwen2.5 0.5B",
        env: Some(("QWEN_MODEL", "qwen2.5-0.5b")),
        download_args: &["--model", "qwen2.5-0.5b"],
        blocked: true,
    },
    ModelInfo {
        id: "qwen-3",
        dir: "qwen",
        name: "Qwen3 0.6B",
        env: Some(("QWEN_MODEL", "qwen3-0.6b")),
        download_args: &["--model", "qwen3-0.6b"],
        blocked: true,
    },
];

fn model_dir(model: &ModelInfo) -> PathBuf {
    PathBuf::from("crates/model-checks").join(model.dir)
}

fn model_envs(model: &ModelInfo) -> Option<HashMap<&str, &str>> {
    model.env.map(|(k, v)| {
        let mut m = HashMap::new();
        m.insert(k, v);
        m
    })
}

fn cargo_args<'a>(base: &'a str, features: &'a str, release: bool) -> Vec<&'a str> {
    let mut args = vec![base, "--no-default-features", "--features", features];
    if release {
        args.push("--release");
    }
    args
}

fn download(model: &ModelInfo) -> anyhow::Result<()> {
    let dir = model_dir(model);
    let mut args = vec!["run", "get_model.py"];
    args.extend_from_slice(model.download_args);
    info!("Downloading {} artifacts...", model.name);
    run_process(
        "uv",
        &args,
        None,
        Some(&dir),
        &format!("Failed to download {} model", model.name),
    )
}

fn build(model: &ModelInfo, features: &str, release: bool) -> anyhow::Result<()> {
    let dir = model_dir(model);
    let envs = model_envs(model);
    let args = cargo_args("build", features, release);
    info!("Building {}...", model.name);
    run_process(
        "cargo",
        &args,
        envs,
        Some(&dir),
        &format!("Failed to build {} model check", model.name),
    )
}

fn run_model(model: &ModelInfo, features: &str, release: bool) -> anyhow::Result<()> {
    let dir = model_dir(model);
    let envs = model_envs(model);
    let args = cargo_args("run", features, release);
    info!("Running {}...", model.name);
    run_process(
        "cargo",
        &args,
        envs,
        Some(&dir),
        &format!("Failed to run {} model check", model.name),
    )
}

fn run_one(
    model: &ModelInfo,
    subcmd: &ModelCheckSubCommand,
    features: &str,
    release: bool,
) -> anyhow::Result<()> {
    match subcmd {
        ModelCheckSubCommand::Download => download(model),
        ModelCheckSubCommand::Build => build(model, features, release),
        ModelCheckSubCommand::Run => run_model(model, features, release),
        ModelCheckSubCommand::All => {
            download(model)?;
            build(model, features, release)?;
            run_model(model, features, release)
        }
    }
}

pub fn handle_command(args: ModelCheckArgs) -> anyhow::Result<()> {
    let subcmd = args.command.unwrap_or(ModelCheckSubCommand::All);
    let release = !args.debug;

    let models: Vec<&ModelInfo> = match &args.model {
        Some(name) => {
            let m = MODELS
                .iter()
                .find(|m| m.id == name.as_str())
                .ok_or_else(|| {
                    let valid: Vec<&str> = MODELS.iter().map(|m| m.id).collect();
                    anyhow::anyhow!(
                        "Unknown model '{}'. Valid models: {}",
                        name,
                        valid.join(", ")
                    )
                })?;
            vec![m]
        }
        None => MODELS.iter().filter(|m| !m.blocked).collect(),
    };

    let mut failed: Vec<&str> = Vec::new();

    for model in &models {
        if let Err(e) = run_one(model, &subcmd, &args.features, release) {
            error!("\x1B[31;1m{} failed: {}\x1B[0m", model.name, e);
            if args.fail_fast {
                return Err(e);
            }
            failed.push(model.name);
        }
    }

    // Summary
    let passed = models.len() - failed.len();
    if failed.is_empty() {
        info!("\x1B[32;1mAll {} model(s) passed\x1B[0m", models.len());
    } else {
        error!(
            "\x1B[31;1m{}/{} passed, {} failed: {}\x1B[0m",
            passed,
            models.len(),
            failed.len(),
            failed.join(", ")
        );
        anyhow::bail!("{} model check(s) failed", failed.len());
    }

    Ok(())
}
