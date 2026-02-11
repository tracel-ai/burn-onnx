#[macro_use]
extern crate log;

mod model_check;

use std::time::Instant;
use tracel_xtask::prelude::*;

// no-std
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";
const ARM_NO_ATOMIC_PTR_TARGET: &str = "thumbv6m-none-eabi";

#[macros::base_commands(
    Bump,
    Check,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Build Burn ONNX in different modes.
    Build(BuildCmdArgs),
    /// Test Burn ONNX.
    Test(TestCmdArgs),
    /// Download, build, and run model checks.
    ModelCheck(model_check::ModelCheckArgs),
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
        Command::Build(cmd_args) => {
            base_commands::build::handle_command(cmd_args, environment, args.context)
        }
        Command::Test(cmd_args) => {
            base_commands::test::handle_command(cmd_args, environment, args.context)
        }
        Command::ModelCheck(cmd_args) => model_check::handle_command(cmd_args),
        _ => dispatch_base_commands(args, environment),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
