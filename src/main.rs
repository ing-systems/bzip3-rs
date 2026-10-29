//! Thin `bzip3` binary with reference behavior parity.
//!
//! Encode, decode, test, and recover modes run over the 9-byte streaming frame.
//! Exit codes, stderr texts, `.bz3` naming rules, and TTY refusal match
//! the C reference CLI. Recover mode differs from the reference: a damaged
//! block gives zero bytes or its checksum-faulty bytes, the exit code is 2,
//! and the run never removes a file. Each input file runs
//! alone, and success removes the input unless `-k`, `-c`, test, or recover mode applies.
//! The default job count uses all physical cores, a recorded
//! divergence from the C default of one thread. Output bytes stay
//! identical at every job count. `clap` generates the help text, a
//! recorded divergence from the reference layout.
//!
//! The binary only wires three modules: [`cli_args`] parses and validates
//! the command line, [`cli_iter`] walks and decodes block batches, and
//! [`cli_run`] executes files and standard streams.

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cognitive_complexity)]

mod cli_args;
mod cli_iter;
mod cli_run;

use std::process::ExitCode;

use cli_args::Stop;
use cli_args::parse_config;
use cli_run::run;

fn main() -> ExitCode {
    match parse_config(std::env::args_os().collect()) {
        Ok(config) => run(&config),
        Err(Stop::Display(error)) => {
            if error.print().is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(Stop::Error(error)) => {
            eprintln!("{error}");
            ExitCode::from(error.code)
        }
    }
}
