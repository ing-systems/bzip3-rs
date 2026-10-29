//! CLI argument handling and block-size handling.
//!
//! This module parses the command line and validates the block size,
//! the worker count, and the mode. It owns [`Config`], [`ExitError`],
//! and [`Stop`]. It has no dependency on the other CLI modules.

// Visibility stays `pub(crate)` for the sibling CLI modules.
// Reason: the parent module is private, so the
// nursery lint asks for plain `pub`, but `pub(crate)` states the
// crate-wide intent at each item.
#![expect(clippy::redundant_pub_crate)]

use std::borrow::Cow;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;

use bzip3_rs::BlockSize;
use bzip3_rs::default_workers;
use clap::ColorChoice;
use clap::Parser;
use clap::error::ErrorKind;

/// Version text. The number is the crate semver, a recorded
/// divergence from the C build macro.
const VERSION_TEXT: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\nCopyright (C) by Kamila Szewczyk, 2022-2025.\n\
    License: GNU Lesser GPL version 3 <https://www.gnu.org/licenses/lgpl-3.0.en.html>"
);

/// Text after the option list. It shows the `-1..-14` flags that `clap` hides.
const AFTER_HELP: &str = "Levels (use -N to set the block size):\n  \
    -1   65 KiB    -6    2 MiB     -11   64 MiB\n  \
    -2  128 KiB    -7    4 MiB     -12  128 MiB\n  \
    -3  256 KiB    -8    8 MiB     -13  256 MiB\n  \
    -4  512 KiB    -9   16 MiB     -14  511 MiB\n  \
    -5    1 MiB    -10  32 MiB\n\n\
    The default level is -9. The last block size flag wins.\n\n\
    Report bugs to: https://github.com/ing-systems/bzip3-rs";

/// Largest valid job count, mirroring the reference range check.
const MAX_WORKERS: usize = 64;

/// Start of the tip line that `clap` adds to an error.
const CLAP_TIP: &str = "\n\nFor more information, try '";

/// CLI mode. The last mode flag on the command line wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Compress input (default).
    Encode,
    /// Decompress input.
    Decode,
    /// Verify input without writing output.
    Test,
    /// Decompress input and keep the data of damaged blocks.
    Recover,
}

/// Program name that selects the default mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Alias {
    /// `bzip3`: encode.
    Bzip3,
    /// `bunzip3`: decode.
    Bunzip3,
    /// `bz3cat`: decode to standard output.
    Bz3cat,
}

impl Alias {
    /// Read the alias from `argv[0]`. Unknown or missing names give `Bzip3`.
    fn from_argv0(argv0: Option<&OsStr>) -> Self {
        let name = argv0
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(OsStr::to_str);
        match name {
            Some("bunzip3") => Self::Bunzip3,
            Some("bz3cat") => Self::Bz3cat,
            _ => Self::Bzip3,
        }
    }

    /// Mode that applies when no mode flag is present.
    const fn default_mode(self) -> Mode {
        match self {
            Self::Bzip3 => Mode::Encode,
            Self::Bunzip3 | Self::Bz3cat => Mode::Decode,
        }
    }
}

/// A `-b` value or a level that passed the syntax check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockSizeArg {
    /// Size inside the valid range.
    Valid(BlockSize),
    /// Number outside the valid range. Fails only when it is the last block flag.
    OutOfRange,
}

// Raw command line. `clap` parses it in one pass.
// Overrides keep only the last mode flag and the last block flag.
// A plain comment keeps this text out of the `--help` output.
// Eight flags mirror the CLI surface one to one, so keep them as bools.
#[expect(clippy::struct_excessive_bools)]
#[derive(Parser)]
#[command(
    name = "bzip3",
    bin_name = "bzip3",
    version = VERSION_TEXT,
    about = "Better and stronger spiritual successor to bzip2.",
    after_help = AFTER_HELP,
    color = ColorChoice::Never,
    args_override_self = true
)]
struct Args {
    /// Compress data (default).
    #[arg(short = 'e', visible_short_alias = 'z', long, overrides_with_all = ["decode", "test", "recover"])]
    encode: bool,
    /// Decompress data.
    #[arg(short = 'd', long, overrides_with_all = ["encode", "test", "recover"])]
    decode: bool,
    /// Verify compressed data.
    #[arg(short = 't', long, overrides_with_all = ["encode", "decode", "recover"])]
    test: bool,
    /// Recover damaged data. Write zeros for bad blocks.
    #[arg(short = 'r', long, overrides_with_all = ["encode", "decode", "test"])]
    recover: bool,
    /// Force standard output.
    #[arg(short = 'c', long)]
    stdout: bool,
    /// Overwrite existing output.
    #[arg(short, long)]
    force: bool,
    /// Keep input files.
    #[arg(short, long)]
    keep: bool,
    /// Print byte counts to standard error.
    #[arg(short, long)]
    verbose: bool,
    /// Suppress non-essential warnings.
    #[arg(short, long)]
    quiet: bool,
    /// Set the block size. Bare N or NM/NMiB is MiB, NK/NKiB is KiB. The default is 16M.
    #[arg(
        short = 'b',
        long = "block",
        value_name = "N",
        value_parser = parse_block_size,
        overrides_with_all = ["level", "fast", "best"]
    )]
    block: Option<BlockSizeArg>,
    /// Block size level. The command line rewrites `-1..-14` to this flag.
    #[arg(
        long,
        hide = true,
        value_name = "N",
        value_parser = parse_level,
        overrides_with_all = ["block", "fast", "best"]
    )]
    level: Option<BlockSizeArg>,
    /// Use the smallest block size, same as -1.
    #[arg(long, overrides_with_all = ["block", "level", "best"])]
    fast: bool,
    /// Use the largest block size, same as -14.
    #[arg(long, overrides_with_all = ["block", "level", "fast"])]
    best: bool,
    /// Set the number of worker threads. 0 uses physical cores.
    #[arg(
        short = 'j',
        long,
        value_name = "N",
        value_parser = parse_jobs,
        default_value = "0"
    )]
    jobs: usize,
    /// Accepted for reference compatibility. Has no effect.
    #[arg(short = 'B', long = "batch", hide = true)]
    _batch: bool,
    /// Input files.
    #[arg(value_name = "files")]
    files: Vec<PathBuf>,
}

impl Args {
    /// Pick the block size. A level, a `-b` value, `--fast`, or `--best`
    /// can remain; the overrides keep only one of them.
    fn block_size(&self) -> Result<BlockSize, ExitError> {
        let chosen = if self.fast {
            Some(BlockSizeArg::Valid(BlockSize::MIN))
        } else if self.best {
            Some(BlockSizeArg::Valid(BlockSize::MAX))
        } else {
            self.level.or(self.block)
        };
        match chosen {
            None => Ok(BlockSize::DEFAULT),
            Some(BlockSizeArg::Valid(size)) => Ok(size),
            Some(BlockSizeArg::OutOfRange) => Err(ExitError::usage(
                "Block size must be between 65 KiB and 511 MiB.",
            )),
        }
    }

    /// Pick the mode. The overrides keep only the last mode flag.
    const fn mode(&self, alias: Alias) -> Mode {
        if self.decode {
            Mode::Decode
        } else if self.test {
            Mode::Test
        } else if self.recover {
            Mode::Recover
        } else if self.encode {
            Mode::Encode
        } else {
            alias.default_mode()
        }
    }
}

/// Validated settings of one run.
// Five flags mirror the CLI surface one to one, so keep them as bools.
#[expect(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub(crate) struct Config {
    /// Selected mode.
    pub(crate) mode: Mode,
    /// Write to standard output (`-c`).
    pub(crate) to_stdout: bool,
    /// Overwrite existing output (`-f`).
    pub(crate) force: bool,
    /// Keep input files (`-k`).
    pub(crate) keep: bool,
    /// Print byte counts to stderr (`-v`).
    pub(crate) verbose: bool,
    /// Suppress non-essential warnings (`-q`).
    pub(crate) quiet: bool,
    /// Block size in bytes.
    pub(crate) block_size: BlockSize,
    /// Worker count, already resolved to 1..=`MAX_WORKERS`.
    pub(crate) workers: usize,
    /// Positional file paths.
    pub(crate) files: Vec<PathBuf>,
}

/// A failure that ends one run, with its exit code.
#[derive(Debug)]
pub(crate) struct ExitError {
    /// Process exit code.
    pub(crate) code: u8,
    /// Text for standard error, without a final newline.
    pub(crate) message: String,
}

impl ExitError {
    /// Build an error with code 1: usage, I/O, and encode faults.
    pub(crate) fn usage(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: message.into(),
        }
    }

    /// Build an error with code 2: faults in compressed data.
    pub(crate) fn corrupt(message: impl Into<String>) -> Self {
        Self {
            code: 2,
            message: message.into(),
        }
    }

    /// Map a codec failure to its message and code.
    ///
    /// I/O failures and failures to build state use code 1. Encode
    /// faults use 1 as well; decode faults of the data use 2. The serial
    /// wording names one block; the parallel wording names the data, like
    /// the reference.
    pub(crate) fn from_codec(error: &bzip3_rs::error::Error, mode: Mode, parallel: bool) -> Self {
        let encode = mode == Mode::Encode;
        match error {
            bzip3_rs::error::Error::Io(io_error) => {
                Self::usage(format!("Write error: {}", os_error_text(io_error)))
            }
            bzip3_rs::error::Error::OutOfMemory
            | bzip3_rs::error::Error::InvalidBlockSize
            | bzip3_rs::error::Error::ThreadPoolFailed => {
                Self::usage("Failed to create a block encoder state.")
            }
            _ => {
                let what = match (encode, parallel) {
                    (true, true) => "Failed to encode data: ",
                    (true, false) => "Failed to encode a block: ",
                    (false, true) => "Failed to decode data: ",
                    (false, false) => "Failed to decode a block: ",
                };
                let message = format!("{what}{error}");
                if encode {
                    Self::usage(message)
                } else {
                    Self::corrupt(message)
                }
            }
        }
    }
}

impl fmt::Display for ExitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Reason the command line ends the run before any file runs.
#[derive(Debug)]
pub(crate) enum Stop {
    /// Print the help or version text of `clap`, exit code 0.
    Display(clap::Error),
    /// Print an error.
    Error(ExitError),
}

impl From<ExitError> for Stop {
    fn from(error: ExitError) -> Self {
        Self::Error(error)
    }
}

impl From<clap::Error> for Stop {
    fn from(error: clap::Error) -> Self {
        use std::error::Error as _;
        match (error.kind(), error.source()) {
            (ErrorKind::DisplayHelp | ErrorKind::DisplayVersion, _) => Self::Display(error),
            // The value parsers word these errors like the reference.
            (ErrorKind::ValueValidation, Some(source)) => {
                Self::Error(ExitError::usage(format!("bzip3: {source}")))
            }
            _ => Self::Error(ExitError::usage(render_usage_error(&error))),
        }
    }
}

/// Render a `clap` error with the reference hint line.
///
/// `clap` adds its own tip because the help flag exists. Replace it
/// with the reference tip.
fn render_usage_error(error: &clap::Error) -> String {
    let text = error.to_string();
    let body = text
        .split_once(CLAP_TIP)
        .map_or_else(|| text.trim_end(), |(body, _)| body);
    format!("{body}\nTry 'bzip3 --help' for more information.")
}

/// Map an OS error to its C `strerror` text for common cases.
pub(crate) fn os_error_text(error: &std::io::Error) -> Cow<'static, str> {
    use std::io::ErrorKind as Kind;
    let text = match error.kind() {
        Kind::NotFound => "No such file or directory",
        Kind::PermissionDenied => "Permission denied",
        Kind::AlreadyExists => "File exists",
        Kind::BrokenPipe => "Broken pipe",
        Kind::IsADirectory => "Is a directory",
        Kind::InvalidInput => "Invalid argument",
        Kind::ReadOnlyFilesystem => "Read-only file system",
        _ => return Cow::Owned(error.to_string()),
    };
    Cow::Borrowed(text)
}

/// Parse a `-b` value. Bare digits and `M`/`MiB` suffixes select MiB;
/// `K`/`KiB` suffixes select KiB. Suffixes ignore case. Returns an
/// error text for malformed values and `OutOfRange` for numeric
/// values outside the valid range.
fn parse_block_size(text: &str) -> Result<BlockSizeArg, String> {
    if text.is_empty() {
        return Ok(BlockSizeArg::OutOfRange);
    }
    let split = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (digits, suffix) = text.split_at(split);
    if digits.is_empty() {
        return Err(format!("invalid block size: {text}"));
    }
    let count = u32::try_from(digits.parse::<u64>().unwrap_or(u64::MAX)).ok();
    let size = match suffix.to_ascii_lowercase().as_str() {
        "" | "m" | "mib" => count.and_then(|mib| BlockSize::mib(mib).ok()),
        "k" | "kib" => count.and_then(|kib| BlockSize::kib(kib).ok()),
        _ => return Err(format!("invalid block size: {text}")),
    };
    Ok(size.map_or(BlockSizeArg::OutOfRange, BlockSizeArg::Valid))
}

/// Parse a level from 1 to 14.
fn parse_level(text: &str) -> Result<BlockSizeArg, String> {
    text.parse::<u8>()
        .ok()
        .and_then(|level| BlockSize::from_level(level).ok())
        .map(BlockSizeArg::Valid)
        .ok_or_else(|| format!("invalid level: {text}"))
}

/// Parse a `-j` value the reference way: every byte is an ASCII digit.
/// The empty text is zero, as in C `atoi`. Large values saturate.
fn parse_jobs(text: &str) -> Result<usize, String> {
    if !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid amount of jobs: {text}"));
    }
    if text.is_empty() {
        return Ok(0);
    }
    Ok(text.parse().unwrap_or(usize::MAX))
}

/// Parse an exact level token `-1..-14`. Returns the level, or
/// `None` for any other token. Digits inside clusters never count.
fn parse_level_token(token: &OsStr) -> Option<u8> {
    let digits = token.to_str()?.strip_prefix('-')?;
    if digits.len() > 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits
        .parse()
        .ok()
        .filter(|&level| BlockSize::from_level(level).is_ok())
}

/// Rewrite each exact level token `-1..-14` to `--level=N`.
///
/// `clap` defines no digit flags, so a cluster like `-19` fails there
/// as an unknown flag. Tokens after `--` stay as they are.
fn rewrite_level_tokens(argv: Vec<OsString>) -> Vec<OsString> {
    let mut literal = false;
    argv.into_iter()
        .enumerate()
        .map(|(index, arg)| {
            if index == 0 || literal {
                return arg;
            }
            if arg == "--" {
                literal = true;
                return arg;
            }
            parse_level_token(&arg).map_or(arg, |level| format!("--level={level}").into())
        })
        .collect()
}

/// Map a CLI job count to a worker count. Zero uses physical
/// cores, limited by available parallelism. Both cases clamp to
/// the reference range.
fn resolve_workers(requested: usize) -> usize {
    let workers = if requested == 0 {
        default_workers()
    } else {
        requested
    };
    workers.min(MAX_WORKERS)
}

/// Parse and validate the command line.
///
/// Help, version, and every error stop here. The first flag in
/// command-line order that fails or stops wins. The last mode flag
/// selects the mode. The last block flag (`-b`, `-1..-14`, `--fast`,
/// `--best`) selects the size; a numeric out-of-range `-b` only fails
/// when it is last. `argv[0]` (`bunzip3`, `bz3cat`) sets the default mode.
pub(crate) fn parse_config(argv: Vec<OsString>) -> Result<Config, Stop> {
    let alias = Alias::from_argv0(argv.first().map(OsString::as_os_str));
    let parsed = Args::try_parse_from(rewrite_level_tokens(argv))?;
    let block_size = parsed.block_size()?;
    if parsed.jobs > MAX_WORKERS {
        return Err(ExitError::usage("Number of workers must be between 0 and 64.").into());
    }
    Ok(Config {
        mode: parsed.mode(alias),
        to_stdout: parsed.stdout || alias == Alias::Bz3cat,
        force: parsed.force,
        keep: parsed.keep,
        verbose: parsed.verbose,
        quiet: parsed.quiet,
        block_size,
        workers: resolve_workers(parsed.jobs),
        files: parsed.files,
    })
}

#[cfg(test)]
mod tests {
    use super::Alias;
    use super::BlockSizeArg;
    use super::Config;
    use super::MAX_WORKERS;
    use super::Mode;
    use super::Stop;
    use clap::error::ErrorKind;

    use super::parse_block_size;
    use super::parse_config;
    use super::parse_level_token;
    use super::resolve_workers;
    use super::rewrite_level_tokens;
    use bzip3_rs::BlockSize;
    use bzip3_rs::default_workers;
    use std::ffi::OsStr;
    use std::ffi::OsString;
    use std::path::PathBuf;

    fn parse(args: &[&str]) -> Result<Config, Stop> {
        parse_config(args.iter().map(OsString::from).collect())
    }

    fn config(args: &[&str]) -> Config {
        parse(args).expect("valid command line")
    }

    fn error_text(args: &[&str]) -> String {
        match parse(args) {
            Err(Stop::Error(error)) => error.message,
            other => panic!("expected an error for {args:?}, got {other:?}"),
        }
    }

    fn level(level: u8) -> BlockSize {
        BlockSize::from_level(level).expect("valid level")
    }

    #[test]
    fn block_values_accept_kib_mib_suffixes() {
        let mib16 = BlockSize::mib(16).unwrap();
        let max = BlockSize::MAX;
        let min = BlockSize::MIN;
        for text in [
            "16", "16M", "16m", "16MiB", "16mib", "16MIB", "16384K", "16384k", "16384KiB",
            "16384kib",
        ] {
            assert_eq!(
                parse_block_size(text),
                Ok(BlockSizeArg::Valid(mib16)),
                "{text}"
            );
        }
        for text in ["65K", "65k", "65KIB", "65kib", "65KiB"] {
            assert_eq!(
                parse_block_size(text),
                Ok(BlockSizeArg::Valid(min)),
                "{text}"
            );
        }
        for text in ["511M", "511mib", "523264K", "523264kib"] {
            assert_eq!(
                parse_block_size(text),
                Ok(BlockSizeArg::Valid(max)),
                "{text}"
            );
        }
    }

    #[test]
    fn block_values_reject_malformed() {
        for text in [
            "xx", "K", "M", "KiB", "-1", "1 ", " 1", "16B", "16MB", "16KB", "16MI", "1.5M", "0x10M",
        ] {
            assert_eq!(
                parse_block_size(text),
                Err(format!("invalid block size: {text}")),
                "{text}"
            );
        }
    }

    #[test]
    fn block_values_flag_out_of_range() {
        for text in [
            "",
            "0",
            "512",
            "64K",
            "523265K",
            "512M",
            "9999999999M",
            "9999999999K",
            "0K",
        ] {
            assert_eq!(
                parse_block_size(text),
                Ok(BlockSizeArg::OutOfRange),
                "{text}"
            );
        }
    }

    #[test]
    fn worker_counts_map_to_pool() {
        assert_eq!(resolve_workers(1), 1);
        assert_eq!(resolve_workers(2), 2);
        assert_eq!(resolve_workers(64), 64);
        assert_eq!(resolve_workers(65), 64);
        assert_eq!(resolve_workers(usize::MAX), 64);
        assert_eq!(resolve_workers(0), default_workers().min(MAX_WORKERS));
        assert!((1..=MAX_WORKERS).contains(&resolve_workers(0)));
    }

    #[test]
    fn alias_selects_default_mode() {
        let alias = |name: &str| Alias::from_argv0(Some(OsStr::new(name)));
        assert_eq!(alias("/usr/bin/bunzip3"), Alias::Bunzip3);
        assert_eq!(alias("bz3cat"), Alias::Bz3cat);
        assert_eq!(alias("bzip3"), Alias::Bzip3);
        assert_eq!(alias("other"), Alias::Bzip3);
        assert_eq!(Alias::from_argv0(None), Alias::Bzip3);
        assert_eq!(config(&["bunzip3"]).mode, Mode::Decode);
        assert_eq!(config(&["bunzip3", "-e"]).mode, Mode::Encode);
        assert_eq!(config(&["bunzip3", "-r"]).mode, Mode::Recover);
        assert_eq!(config(&["bz3cat", "--recover"]).mode, Mode::Recover);
        assert!(config(&["bz3cat"]).to_stdout);
    }

    #[test]
    fn last_mode_flag_wins() {
        assert_eq!(config(&["bzip3"]).mode, Mode::Encode);
        assert_eq!(config(&["bzip3", "-e", "-d"]).mode, Mode::Decode);
        assert_eq!(config(&["bzip3", "-d", "-t"]).mode, Mode::Test);
        assert_eq!(config(&["bzip3", "-d", "-z"]).mode, Mode::Encode);
        assert_eq!(config(&["bzip3", "-d", "-d"]).mode, Mode::Decode);
        assert_eq!(config(&["bzip3", "-d", "-r"]).mode, Mode::Recover);
        assert_eq!(config(&["bzip3", "-r", "-t"]).mode, Mode::Test);
        assert_eq!(config(&["bzip3", "-r", "-e"]).mode, Mode::Encode);
    }

    #[test]
    fn repeated_flags_are_accepted() {
        let parsed = config(&["bzip3", "-k", "-k", "-j", "2", "-j", "3"]);
        assert!(parsed.keep);
        assert_eq!(parsed.workers, 3);
    }

    #[test]
    fn help_and_version_follow_argument_order() {
        assert!(matches!(
            parse(&["bzip3", "-h", "-b", "xx"]),
            Err(Stop::Display(ref e)) if e.kind() == ErrorKind::DisplayHelp
        ));
        assert!(matches!(
            parse(&["bzip3", "-h", "--bogus"]),
            Err(Stop::Display(ref e)) if e.kind() == ErrorKind::DisplayHelp
        ));
        assert!(matches!(
            parse(&["bzip3", "-V", "-j", "xx"]),
            Err(Stop::Display(ref e)) if e.kind() == ErrorKind::DisplayVersion
        ));
        assert_eq!(
            error_text(&["bzip3", "-b", "xx", "-h"]),
            "bzip3: invalid block size: xx"
        );
        assert_eq!(
            error_text(&["bzip3", "-j", "xx", "--version"]),
            "bzip3: invalid amount of jobs: xx"
        );
    }

    #[test]
    fn last_block_flag_wins() {
        assert_eq!(config(&["bzip3"]).block_size, BlockSize::DEFAULT);
        assert_eq!(config(&["bzip3", "-b", "1", "-14"]).block_size, level(14));
        assert_eq!(
            config(&["bzip3", "-14", "-b", "1"]).block_size,
            BlockSize::mib(1).unwrap()
        );
        assert_eq!(config(&["bzip3", "--fast", "--best"]).block_size, level(14));
        assert_eq!(config(&["bzip3", "--best", "--fast"]).block_size, level(1));
        assert_eq!(config(&["bzip3", "-1", "-14"]).block_size, level(14));
        assert_eq!(config(&["bzip3", "-14", "-1"]).block_size, level(1));
        assert_eq!(config(&["bzip3", "-b", "512", "-14"]).block_size, level(14));
        assert_eq!(
            config(&["bzip3", "-b", "1", "--", "-14"]).files,
            [PathBuf::from("-14")]
        );
        assert_eq!(
            error_text(&["bzip3", "-14", "-b", "512"]),
            "Block size must be between 65 KiB and 511 MiB."
        );
        assert_eq!(
            error_text(&["bzip3", "-b", "xx", "-14"]),
            "bzip3: invalid block size: xx"
        );
    }

    #[test]
    fn jobs_follow_reference_rules() {
        assert_eq!(config(&["bzip3", "-j", "1"]).workers, 1);
        assert_eq!(config(&["bzip3", "-j", "64"]).workers, 64);
        assert!(config(&["bzip3", "-j", ""]).workers >= 1);
        assert_eq!(
            error_text(&["bzip3", "-j", "65"]),
            "Number of workers must be between 0 and 64."
        );
        for text in ["+1", "1x"] {
            assert_eq!(
                error_text(&["bzip3", "-j", text]),
                format!("bzip3: invalid amount of jobs: {text}")
            );
        }
        // A level token never serves as a value.
        let text = error_text(&["bzip3", "-j", "-1"]);
        assert!(text.starts_with("error: a value is required"), "{text}");
    }

    #[test]
    fn digit_clusters_stay_unknown_flags() {
        for flag in ["-19", "-0", "-15", "-91"] {
            let text = error_text(&["bzip3", flag]);
            assert!(text.starts_with("error: unexpected argument"), "{text}");
        }
    }

    #[test]
    fn usage_errors_show_one_hint() {
        let text = error_text(&["bzip3", "--bogus"]);
        assert!(text.contains("Usage: bzip3"), "{text}");
        assert!(
            text.ends_with("\nTry 'bzip3 --help' for more information."),
            "{text}"
        );
        assert_eq!(text.matches("Try '").count(), 1, "{text}");
        assert!(!text.contains("For more information"), "{text}");
    }

    #[test]
    fn from_codec_covers_all_arms() {
        use super::ExitError;
        use bzip3_rs::error::Error;
        let io = Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        let fault = ExitError::from_codec(&io, Mode::Encode, false);
        assert_eq!(fault.code, 1);
        assert_eq!(fault.message, "Write error: No such file or directory");
        for state in [
            Error::OutOfMemory,
            Error::InvalidBlockSize,
            Error::ThreadPoolFailed,
        ] {
            let fault = ExitError::from_codec(&state, Mode::Encode, true);
            assert_eq!(fault.code, 1);
            assert_eq!(fault.message, "Failed to create a block encoder state.");
        }
        let fault = ExitError::from_codec(&Error::BwtFailed, Mode::Encode, false);
        assert_eq!(fault.code, 1);
        assert!(
            fault.message.starts_with("Failed to encode a block: "),
            "{fault}"
        );
        let fault = ExitError::from_codec(&Error::BwtFailed, Mode::Encode, true);
        assert_eq!(fault.code, 1);
        assert!(
            fault.message.starts_with("Failed to encode data: "),
            "{fault}"
        );
        let fault = ExitError::from_codec(&Error::TruncatedData, Mode::Decode, false);
        assert_eq!(fault.code, 2);
        assert!(
            fault.message.starts_with("Failed to decode a block: "),
            "{fault}"
        );
        let fault = ExitError::from_codec(&Error::TruncatedData, Mode::Decode, true);
        assert_eq!(fault.code, 2);
        assert!(
            fault.message.starts_with("Failed to decode data: "),
            "{fault}"
        );
        for mode in [Mode::Test, Mode::Recover] {
            let fault = ExitError::from_codec(&Error::TruncatedData, mode, true);
            assert_eq!(fault.code, 2);
            assert!(
                fault.message.starts_with("Failed to decode data: "),
                "{fault}"
            );
        }
        let shown = format!(
            "{}",
            ExitError::from_codec(&Error::BwtFailed, Mode::Encode, false)
        );
        assert!(shown.starts_with("Failed to encode a block: "), "{shown}");
    }

    #[test]
    fn os_error_text_covers_known_and_fallback() {
        use super::os_error_text;
        use std::io::ErrorKind as Kind;
        for (kind, text) in [
            (Kind::NotFound, "No such file or directory"),
            (Kind::PermissionDenied, "Permission denied"),
            (Kind::AlreadyExists, "File exists"),
            (Kind::BrokenPipe, "Broken pipe"),
            (Kind::IsADirectory, "Is a directory"),
            (Kind::InvalidInput, "Invalid argument"),
            (Kind::ReadOnlyFilesystem, "Read-only file system"),
        ] {
            assert_eq!(os_error_text(&std::io::Error::new(kind, "x")), text);
        }
        let other = std::io::Error::other("custom boom");
        assert_eq!(os_error_text(&other), "custom boom");
    }

    #[cfg(unix)]
    fn non_utf8_level_tokens() -> Vec<OsString> {
        use std::os::unix::ffi::OsStrExt as _;
        vec![
            OsStr::from_bytes(b"\xff\xfe").to_owned(),
            OsStr::from_bytes(b"-\xff").to_owned(),
        ]
    }

    #[cfg(windows)]
    fn non_utf8_level_tokens() -> Vec<OsString> {
        use std::os::windows::ffi::OsStringExt as _;
        vec![
            OsString::from_wide(&[0xD800]),
            OsString::from_wide(&[u16::from(b'-'), 0xD800]),
        ]
    }

    #[cfg(not(any(unix, windows)))]
    fn non_utf8_level_tokens() -> Vec<OsString> {
        Vec::new()
    }

    #[test]
    fn parse_level_token_covers_edges() {
        assert_eq!(parse_level_token(OsStr::new("-1")), Some(1));
        assert_eq!(parse_level_token(OsStr::new("-14")), Some(14));
        assert_eq!(parse_level_token(OsStr::new("-01")), Some(1));
        for text in ["", "-", "--1", "-0", "-15", "-19", "1", "-1a", "-x", "--"] {
            assert_eq!(parse_level_token(OsStr::new(text)), None, "{text}");
        }
        for raw in non_utf8_level_tokens() {
            assert_eq!(parse_level_token(&raw), None);
        }
    }

    #[test]
    fn rewrite_level_tokens_covers_dashdash() {
        let argv =
            |items: &[&str]| rewrite_level_tokens(items.iter().map(OsString::from).collect());
        assert_eq!(
            argv(&["bzip3", "-1"]),
            [OsString::from("bzip3"), OsString::from("--level=1")]
        );
        assert_eq!(
            argv(&["bzip3", "-14", "a", "-2"]),
            [
                OsString::from("bzip3"),
                OsString::from("--level=14"),
                OsString::from("a"),
                OsString::from("--level=2")
            ]
        );
        assert_eq!(
            argv(&["-1", "-2"]),
            [OsString::from("-1"), OsString::from("--level=2")]
        );
        assert_eq!(
            argv(&["bzip3", "--", "-1"]),
            [
                OsString::from("bzip3"),
                OsString::from("--"),
                OsString::from("-1")
            ]
        );
        assert_eq!(
            argv(&["bzip3", "-b", "xx"]),
            [
                OsString::from("bzip3"),
                OsString::from("-b"),
                OsString::from("xx")
            ]
        );
    }
}
