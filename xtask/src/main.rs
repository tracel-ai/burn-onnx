#[macro_use]
extern crate log;

mod check;
mod diff_expectations;
mod expectations_schema;
mod model_check;
mod refresh_onnx_tests;
mod retriage;
mod update_expectations;

use std::time::Instant;
use tracel_xtask::prelude::*;

// no-std
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";
const ARM_NO_ATOMIC_PTR_TARGET: &str = "thumbv6m-none-eabi";

#[derive(clap::Subcommand, strum::Display)]
pub enum Command {
    Bump(BumpCmdArgs),
    Check(CheckCmdArgs),
    Compile(CompileCmdArgs),
    Coverage(CoverageCmdArgs),
    Dependencies(DependenciesCmdArgs),
    Doc(DocCmdArgs),
    Fix(FixCmdArgs),
    Publish(PublishCmdArgs),
    Validate(ValidateCmdArgs),
    Vulnerabilities(VulnerabilitiesCmdArgs),
    /// Build Burn ONNX in different modes.
    Build(BuildCmdArgs),
    /// Test Burn ONNX.
    Test(TestCmdArgs),
    /// Download, build, and run model checks.
    ModelCheck(model_check::ModelCheckArgs),
    /// Refresh the vendored upstream ONNX backend node tests in
    /// `crates/onnx-official-tests/vendor/node/` from a given onnx
    /// release tag.
    RefreshOnnxTests(refresh_onnx_tests::RefreshOnnxTestsArgs),
    /// Compare `crates/onnx-official-tests/expectations.toml` against a
    /// git ref (default `origin/main`) and summarise promotions,
    /// regressions, sideways changes, adds, and removes. Used by CI
    /// to post a PR-comment delta; also useful locally for triage.
    DiffExpectations(diff_expectations::DiffExpectationsArgs),
    /// Run the onnx-official-tests suite and rewrite
    /// `expectations.toml` in place to demote any pass-listed tests
    /// that now fail. Supports `--dry-run` for preview mode.
    UpdateExpectations(update_expectations::UpdateExpectationsArgs),
    /// Re-run the `skip-codegen` and `skip-compile` rows of
    /// `expectations.toml` against the current tree and rewrite them to
    /// match reality. Those rows are never exercised by the build, so
    /// they go stale as the bugs behind them get fixed.
    Retriage(retriage::RetriageArgs),
}

fn dispatch_base_commands(args: XtaskArgs<Command>, env: Environment) -> anyhow::Result<()> {
    match args.command {
        Command::Bump(cmd) => base_commands::bump::handle_command(cmd, env, args.context),
        Command::Check(cmd) => base_commands::check::handle_command(cmd, env, args.context),
        Command::Compile(cmd) => base_commands::compile::handle_command(cmd, env, args.context),
        Command::Coverage(cmd) => base_commands::coverage::handle_command(cmd, env, args.context),
        Command::Dependencies(cmd) => {
            base_commands::dependencies::handle_command(cmd, env, args.context)
        }
        Command::Doc(cmd) => base_commands::doc::handle_command(cmd, env, args.context),
        Command::Fix(cmd) => base_commands::fix::handle_command(cmd, env, args.context, None),
        Command::Publish(cmd) => base_commands::publish::handle_command(cmd, env, args.context),
        Command::Validate(cmd) => base_commands::validate::handle_command(cmd, env, args.context),
        Command::Vulnerabilities(cmd) => {
            base_commands::vulnerabilities::handle_command(cmd, env, args.context)
        }
        _ => Err(anyhow::anyhow!("Unknown command")),
    }
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let (args, environment) = init_xtask::<Command>(parse_args::<Command>()?)?;

    if args.context == Context::NoStd {
        // Install additional targets for no-std execution environments
        rustup_add_target(WASM32_TARGET)?;
        rustup_add_target(ARM_TARGET)?;
        rustup_add_target(ARM_NO_ATOMIC_PTR_TARGET)?;
    }

    match args.command {
        Command::Check(cmd_args) => check::handle_command(cmd_args, environment, args.context),
        Command::Build(cmd_args) => {
            base_commands::build::handle_command(cmd_args, environment, args.context)
        }
        Command::Test(cmd_args) => {
            base_commands::test::handle_command(cmd_args, environment, args.context)
        }
        Command::ModelCheck(cmd_args) => model_check::handle_command(cmd_args),
        Command::RefreshOnnxTests(cmd_args) => refresh_onnx_tests::handle_command(cmd_args),
        Command::DiffExpectations(cmd_args) => diff_expectations::handle_command(cmd_args),
        Command::UpdateExpectations(cmd_args) => update_expectations::handle_command(cmd_args),
        Command::Retriage(cmd_args) => retriage::handle_command(cmd_args),
        _ => dispatch_base_commands(args, environment),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
