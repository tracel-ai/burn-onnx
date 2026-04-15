//! `cargo xtask refresh-onnx-tests --version <tag>` — scripted refresh
//! of the vendored upstream ONNX backend node tests under
//! `crates/onnx-official-tests/vendor/node/`.
//!
//! This command is the intended successor to the manual
//! "download-tarball-and-rsync" steps described in the M1 scaffold's
//! README. It does not reconcile `expectations.toml` automatically;
//! instead it downloads the requested tag, replaces the vendored
//! `node/` tree, and prints a reminder to re-run the tests so that
//! any drift between the new vendor set and the declared
//! expectations surfaces loudly.
//!
//! The reconciliation is intentionally left as an iterative workflow
//! rather than a one-shot script: the first `cargo test` run after a
//! refresh reveals which previously-passing tests now fail (demote
//! to fail-compare), which previously-skipped tests now pass (promote
//! to pass), and which test directories are new or removed (drift
//! check in `tests/test_mod.rs` fails). Trying to automate that would
//! either guess wrong or duplicate the classification logic already
//! encoded in the test runner.

use std::path::{Path, PathBuf};

use tracel_xtask::prelude::*;

/// Arguments for the `refresh-onnx-tests` subcommand.
#[derive(clap::Args)]
pub struct RefreshOnnxTestsArgs {
    /// ONNX release tag to download, e.g. `1.19.0`. The prefix `v` is
    /// optional; both `1.19.0` and `v1.19.0` are accepted.
    #[arg(long)]
    pub version: String,

    /// Skip the actual file-system mutation and just print what would
    /// be done. Useful for verifying the download URL and extraction
    /// target before clobbering the current vendor directory.
    #[arg(long)]
    pub dry_run: bool,
}

/// Top-level dispatcher called from `main.rs`.
pub fn handle_command(args: RefreshOnnxTestsArgs) -> anyhow::Result<()> {
    let version = args.version.trim_start_matches('v');
    let url = format!("https://github.com/onnx/onnx/archive/refs/tags/v{version}.tar.gz");

    let tmp = std::env::temp_dir().join(format!("burn-onnx-refresh-{version}"));
    let tarball = tmp.join(format!("onnx-{version}.tar.gz"));
    let extract_root = tmp.join("extract");
    let extract_src = extract_root
        .join(format!("onnx-{version}"))
        .join("onnx/backend/test/data/node");

    let vendor_root = repo_vendor_dir();

    info!("ONNX version:     v{version}");
    info!("Download URL:     {url}");
    info!("Working dir:      {}", tmp.display());
    info!("Vendor target:    {}", vendor_root.display());

    if args.dry_run {
        warn!("--dry-run set; no files will be modified");
        return Ok(());
    }

    std::fs::create_dir_all(&tmp)?;
    std::fs::create_dir_all(&extract_root)?;

    info!("Downloading tarball...");
    run_process(
        "curl",
        &[
            "-sSL",
            "-o",
            tarball.to_str().expect("non-utf8 tarball path"),
            url.as_str(),
        ],
        None,
        None,
        &format!("Failed to download onnx v{version}"),
    )?;

    info!("Extracting onnx/backend/test/data/node...");
    run_process(
        "tar",
        &[
            "-xzf",
            tarball.to_str().expect("non-utf8 tarball path"),
            "-C",
            extract_root.to_str().expect("non-utf8 extract path"),
            &format!("onnx-{version}/onnx/backend/test/data/node"),
        ],
        None,
        None,
        "tar extraction failed",
    )?;

    if !extract_src.is_dir() {
        return Err(anyhow::anyhow!(
            "expected extracted directory {} does not exist",
            extract_src.display()
        ));
    }

    info!("Replacing vendor/node with extracted data...");
    // We remove the old vendor dir before rsync so deleted upstream
    // tests are reflected instead of accumulating. The rsync is a
    // no-op over identical files, so re-running the command is safe.
    if vendor_root.exists() {
        std::fs::remove_dir_all(&vendor_root)?;
    }
    std::fs::create_dir_all(&vendor_root)?;

    run_process(
        "rsync",
        &[
            "-a",
            &format!("{}/", extract_src.display()),
            &format!("{}/", vendor_root.display()),
        ],
        None,
        None,
        "rsync of extracted data into vendor/node failed",
    )?;

    let count = count_test_dirs(&vendor_root)?;
    info!(
        "Vendored {count} test_* directories into {}",
        vendor_root.display()
    );

    warn!(
        "Next step: run `cargo test -p onnx-official-tests`. \
         The drift check will list any tests that appear in vendor/ \
         but not in expectations.toml (and vice versa); demote any \
         previously-passing tests that now fail, or add pass entries \
         for newly added tests."
    );

    Ok(())
}

/// Absolute path to `crates/onnx-official-tests/vendor/node` based on
/// `CARGO_MANIFEST_DIR` (which for xtask resolves to `<repo>/xtask`).
fn repo_vendor_dir() -> PathBuf {
    let xtask_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `.parent()` climbs from `<repo>/xtask` to `<repo>`; an
    // unwrap_or fallback is unnecessary because xtask always lives at
    // the workspace root.
    let repo_root = xtask_manifest
        .parent()
        .expect("xtask crate has no parent directory")
        .to_path_buf();
    repo_root.join("crates/onnx-official-tests/vendor/node")
}

/// Count subdirectories named `test_*` inside `dir`. Used for a
/// sanity-check summary after the rsync completes so the user can
/// eyeball whether the vendor step landed approximately the expected
/// number of tests (~1600 for v1.19.0).
fn count_test_dirs(dir: &Path) -> anyhow::Result<usize> {
    let mut count = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry
                .file_name()
                .to_str()
                .map(|n| n.starts_with("test_"))
                .unwrap_or(false)
        {
            count += 1;
        }
    }
    Ok(count)
}
