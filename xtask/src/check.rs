//! Project-specific extensions to the standard check commands.

use tracel_xtask::prelude::*;

/// Run the standard checks and lint the opt-in export feature when applicable.
pub fn handle_command(
    args: CheckCmdArgs,
    environment: Environment,
    context: Context,
) -> anyhow::Result<()> {
    let lint_export = matches!(
        args.get_command(),
        CheckSubCommand::Lint | CheckSubCommand::All
    );
    base_commands::check::handle_command(args, environment, context)?;

    if lint_export {
        group!("Lint burn-onnx export feature");
        run_process(
            "cargo",
            &[
                "clippy",
                "--no-deps",
                "--color=always",
                "-p",
                "burn-onnx",
                "--no-default-features",
                "--features",
                "export",
                "--tests",
                "--",
                "--deny",
                "warnings",
            ],
            None,
            None,
            "burn-onnx export lint failed",
        )?;
        endgroup!();
    }

    Ok(())
}
