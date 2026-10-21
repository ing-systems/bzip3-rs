//! CLI integration tests for the `bzip3` binary.
//!
//! Run the built binary in a scratch directory.
//! Assert bytes, exit codes, and stderr text.

#[path = "cli/batch_jobs.rs"]
mod batch_jobs;
#[path = "cli/coverage_extra.rs"]
mod coverage_extra;
#[path = "cli/errors.rs"]
mod errors;
#[path = "cli/levels.rs"]
mod levels;
#[path = "cli/recover.rs"]
mod recover;
#[path = "cli/roundtrip.rs"]
mod roundtrip;
#[path = "cli/support.rs"]
mod support;
#[path = "cli/verbose.rs"]
mod verbose;
