use std::collections::HashMap;
use std::path::PathBuf;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct ModelCheckArgs {
    /// Model to operate on (default: all models).
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
    dir: &'static str,
    name: &'static str,
    /// Optional (env_var, default_value) for models with selectable variants.
    env: Option<(&'static str, &'static str)>,
    /// Skipped in default "all" runs; still runnable with `--model <name>`.
    blocked: bool,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        dir: "silero-vad",
        name: "Silero VAD",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "all-minilm-l6-v2",
        name: "all-MiniLM-L6-v2",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "clip-vit-b-32-text",
        name: "CLIP ViT-B-32 text",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "clip-vit-b-32-vision",
        name: "CLIP ViT-B-32 vision",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "modernbert-base",
        name: "ModernBERT-base",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "rf-detr",
        name: "RF-DETR Small",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "depth-anything-v2",
        name: "Depth-Anything-v2",
        env: None,
        blocked: false,
    },
    ModelInfo {
        dir: "albert",
        name: "ALBERT",
        env: Some(("ALBERT_MODEL", "albert-base-v2")),
        blocked: false,
    },
    ModelInfo {
        dir: "yolo",
        name: "YOLO",
        env: Some(("YOLO_MODEL", "yolov8n")),
        blocked: false,
    },
    ModelInfo {
        dir: "mediapipe-face-detector",
        name: "MediaPipe Face Detector",
        env: None,
        blocked: false,
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
    info!("Downloading {} artifacts...", model.name);
    run_process(
        "uv",
        &["run", "get_model.py"],
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
    let features = &args.features;
    let release = !args.debug;

    let models: Vec<&ModelInfo> = match &args.model {
        Some(name) => {
            let m = MODELS
                .iter()
                .find(|m| m.dir == name.as_str())
                .ok_or_else(|| {
                    let valid: Vec<&str> = MODELS.iter().map(|m| m.dir).collect();
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
        if let Err(e) = run_one(model, &subcmd, features, release) {
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
