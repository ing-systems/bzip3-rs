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

#![forbid(unsafe_code)]
#![deny(clippy::all, clippy::pedantic)]
#![warn(clippy::nursery)]

use std::borrow::Cow;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::fs::Metadata;
use std::io::IsTerminal;
use std::io::Read;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use bzip3_rs::BlockDecoder;
use bzip3_rs::BlockHeader;
use bzip3_rs::BlockSize;
use bzip3_rs::BlockStatus;
use bzip3_rs::Jobs;
use bzip3_rs::StreamHeader;
use bzip3_rs::default_workers;
use bzip3_rs::error::Error;
use bzip3_rs::stream;
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
enum Mode {
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
struct Config {
    /// Selected mode.
    mode: Mode,
    /// Write to standard output (`-c`).
    to_stdout: bool,
    /// Overwrite existing output (`-f`).
    force: bool,
    /// Keep input files (`-k`).
    keep: bool,
    /// Print byte counts to stderr (`-v`).
    verbose: bool,
    /// Suppress non-essential warnings (`-q`).
    quiet: bool,
    /// Block size in bytes.
    block_size: BlockSize,
    /// Worker count, already resolved to 1..=`MAX_WORKERS`.
    workers: usize,
    /// Positional file paths.
    files: Vec<PathBuf>,
}

/// A failure that ends one run, with its exit code.
#[derive(Debug)]
struct ExitError {
    /// Process exit code.
    code: u8,
    /// Text for standard error, without a final newline.
    message: String,
}

impl ExitError {
    /// Build an error with code 1: usage, I/O, and encode faults.
    fn usage(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: message.into(),
        }
    }

    /// Build an error with code 2: faults in compressed data.
    fn corrupt(message: impl Into<String>) -> Self {
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
    fn from_codec(error: &Error, mode: Mode, parallel: bool) -> Self {
        let encode = mode == Mode::Encode;
        match error {
            Error::Io(io_error) => Self::usage(format!("Write error: {}", os_error_text(io_error))),
            Error::OutOfMemory | Error::InvalidBlockSize | Error::ThreadPoolFailed => {
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
enum Stop {
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
fn os_error_text(error: &std::io::Error) -> Cow<'static, str> {
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
fn parse_config(argv: Vec<OsString>) -> Result<Config, Stop> {
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

/// Refuse binary data on a terminal, mirroring the reference:
/// encode refuses binary output to a terminal, decode and test refuse
/// binary input from a terminal.
fn check_terminal(
    mode: Mode,
    stdin_terminal: bool,
    stdout_terminal: bool,
) -> Result<(), ExitError> {
    let refused = match mode {
        Mode::Encode => stdout_terminal,
        Mode::Decode | Mode::Test | Mode::Recover => stdin_terminal,
    };
    if refused {
        return Err(ExitError::usage(
            "Refusing to read/write binary data from/to the terminal.",
        ));
    }
    Ok(())
}

/// Format one verbose line. Byte-identical to the reference format,
/// including the leading space, the tab, and the `OK, ` test marker.
fn format_verbose_line(mode: Mode, file: Option<&str>, read: u64, written: u64) -> String {
    let label = file.map_or_else(String::new, |name| format!(" {name}:"));
    let ok = if mode == Mode::Test { "OK, " } else { "" };
    // The ratio puts the coded size on top: the output size when
    // encoding, the input size otherwise. A zero denominator prints
    // `inf`, exactly like the C `%.2f` output. Precision loss on huge
    // counts matches the reference, which converts to `double` as well.
    let (coded, plain) = if mode == Mode::Encode {
        (written, read)
    } else {
        (read, written)
    };
    #[expect(clippy::cast_precision_loss)]
    let (percent, bits) = (
        coded as f64 * 100.0 / plain as f64,
        coded as f64 * 8.0 / plain as f64,
    );
    format!("{label}\t{ok}{read} -> {written} bytes, {percent:.2}%, {bits:.2} bpb\n")
}

/// Open a positional input file. Returns the file and its metadata.
fn open_input(path: &Path, name: &str) -> Result<(File, Metadata), ExitError> {
    let directory = || ExitError::usage(format!("Error: input `{name}' is a directory."));
    let opened = File::open(path).and_then(|file| file.metadata().map(|meta| (file, meta)));
    match opened {
        Ok((_, metadata)) if metadata.is_dir() => Err(directory()),
        Ok(opened) => Ok(opened),
        Err(_) if path.is_dir() => Err(directory()),
        Err(error) => Err(ExitError::usage(format!(
            "Error: failed to open input file `{name}': {}",
            os_error_text(&error)
        ))),
    }
}

/// Create an output file. An existing file needs `force`.
fn open_output(path: &Path, force: bool) -> Result<File, ExitError> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    options.open(path).map_err(|error| {
        let name = path.display();
        ExitError::usage(if path.is_dir() {
            format!("Error: output file `{name}' is a directory.")
        } else if error.kind() == std::io::ErrorKind::AlreadyExists {
            format!("Error: output file `{name}' already exists. Use -f to force overwrite.")
        } else {
            format!(
                "Error: failed to open output file `{name}': {}",
                os_error_text(&error)
            )
        })
    })
}

/// Read the full input.
fn read_all(input: &mut dyn Read) -> Result<Vec<u8>, ExitError> {
    let mut data = Vec::new();
    input
        .read_to_end(&mut data)
        .map_err(|error| ExitError::usage(format!("Read error: {}", os_error_text(&error))))?;
    Ok(data)
}

/// Flush the output and print the verbose line.
/// Verbose prints only on success.
fn finish(
    output: &mut dyn Write,
    config: &Config,
    file: Option<&str>,
    read: usize,
    written: u64,
) -> Result<(), ExitError> {
    output.flush().map_err(|error| {
        ExitError::usage(format!(
            "Error: Failed on fflush: {}",
            os_error_text(&error)
        ))
    })?;
    if config.verbose {
        eprint!(
            "{}",
            format_verbose_line(config.mode, file, read as u64, written)
        );
    }
    Ok(())
}

/// Run the encode path.
fn encode(
    config: &Config,
    input: &mut dyn Read,
    output: &mut dyn Write,
    file: Option<&str>,
) -> Result<(), ExitError> {
    let data = read_all(input)?;
    // One worker runs serially. More workers run on a private pool
    // that the library builds for inputs with two blocks or more.
    let jobs = Jobs::from_count(config.workers);
    let written = stream::compress_slice(&data, &mut *output, config.block_size, jobs)
        .map_err(|error| ExitError::from_codec(&error, Mode::Encode, config.workers > 1))?;
    finish(output, config, file, data.len(), written)
}

/// Parse the frame header. Returns the block size.
/// Corrupt shapes (bad signature, truncated header, bad block size)
/// use code 2, like the reference data-error path. Recover mode
/// replaces a bad block size with a size from the block headers.
/// The reference uses the largest block size.
fn parse_stream_header(data: &[u8], recover: bool) -> Result<BlockSize, ExitError> {
    if !data.starts_with(&bzip3_rs::MAGIC_NUMBER) {
        return Err(ExitError::corrupt("Invalid signature."));
    }
    let size = data
        .get(bzip3_rs::MAGIC_NUMBER.len()..)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or_else(|| ExitError::corrupt("Error: Corrupt file"))?;
    // A negative wire size reads as a value above the largest block size.
    match BlockSize::new(u32::from_le_bytes(*size)) {
        Ok(block_size) => Ok(block_size),
        Err(_) if recover => {
            eprintln!("Recovery mode: Proceeding.");
            Ok(infer_block_size(data))
        }
        Err(_) => Err(ExitError::corrupt(
            "The input file is corrupted. Reason: Invalid block size in the header.",
        )),
    }
}

/// Find a block size that fits every block header before the first fault.
///
/// The size is the largest original size or payload size of those
/// blocks, limited to the valid range. One pass reads the headers.
/// `data` must hold the frame header.
fn infer_block_size(data: &[u8]) -> BlockSize {
    let largest = BlockSpans::new(data, BlockSize::MAX)
        .map_while(Result::ok)
        .map(|span| {
            let payload = u32::try_from(span.payload.len()).unwrap_or(u32::MAX);
            span.original_size.max(payload)
        })
        .max()
        .unwrap_or(0);
    let size = largest.clamp(BlockSize::MIN.get(), BlockSize::MAX.get());
    BlockSize::new(size).unwrap_or(BlockSize::MAX)
}

/// One parsed block: claimed original size plus payload range.
struct BlockSpan {
    /// Offset of the block header inside the input.
    offset: usize,
    /// Claimed original size.
    original_size: u32,
    /// Byte range of the payload inside the input.
    payload: Range<usize>,
}

/// Walk the block list of a frame, one `BlockSpan` per block.
///
/// A fault ends the walk after it yields the error once.
struct BlockSpans<'a> {
    /// Whole input, frame header included.
    data: &'a [u8],
    /// Offset of the next block header.
    pos: usize,
    /// Largest payload size of one block.
    max_payload: u32,
    /// Largest original size of one block.
    max_original: u32,
    /// Offset of the block header that caused the fault.
    fault_at: usize,
}

impl<'a> BlockSpans<'a> {
    /// Start the walk behind the frame header.
    fn new(data: &'a [u8], block_size: BlockSize) -> Self {
        Self {
            data,
            pos: StreamHeader::LEN,
            max_payload: u32::try_from(bzip3_rs::compress_bound(block_size.as_usize()))
                .unwrap_or(u32::MAX),
            max_original: block_size.get(),
            fault_at: 0,
        }
    }

    /// Parse the block header at `self.pos`. Truncated shapes and
    /// inconsistent sizes use code 2, like the reference data-error path.
    fn parse(&self) -> Result<BlockSpan, ExitError> {
        let corrupt = || ExitError::corrupt("Error: Corrupt file");
        let rest = &self.data[self.pos..];
        let raw = rest
            .first_chunk::<{ BlockHeader::LEN }>()
            .ok_or_else(corrupt)?;
        // A negative wire size reads as a value above both limits.
        let header = BlockHeader::from_bytes(raw);
        if header.compressed_size > self.max_payload || header.original_size > self.max_original {
            return Err(ExitError::corrupt(
                "Failed to decode a block: Inconsistent headers.",
            ));
        }
        let start = self.pos + BlockHeader::LEN;
        let end = start + header.compressed_size as usize;
        if end > self.data.len() {
            return Err(corrupt());
        }
        Ok(BlockSpan {
            offset: self.pos,
            original_size: header.original_size,
            payload: start..end,
        })
    }
}

impl Iterator for BlockSpans<'_> {
    type Item = Result<BlockSpan, ExitError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.data.len() {
            return None;
        }
        let item = self.parse();
        if item.is_err() {
            self.fault_at = self.pos;
        }
        self.pos = item
            .as_ref()
            .map_or(self.data.len(), |span| span.payload.end);
        Some(item)
    }
}

/// One decoder state with the plain bytes and the result of its last block.
struct DecodeSlot {
    /// Decoder of this slot.
    decoder: BlockDecoder,
    /// Plain bytes of the last block.
    output: Vec<u8>,
    /// Result of the last block.
    result: Result<BlockStatus, Error>,
}

impl DecodeSlot {
    /// Build a slot with a new decoder.
    fn new(block_size: BlockSize) -> Result<Self, Error> {
        Ok(Self {
            decoder: BlockDecoder::new(block_size)?,
            output: Vec::new(),
            result: Ok(BlockStatus::Intact),
        })
    }

    /// Decode one block into `self.output` and record the result.
    ///
    /// Recover mode keeps the bytes of a damaged block in `self.output`.
    fn decode(&mut self, data: &[u8], span: &BlockSpan, recover: bool) {
        self.output.clear();
        let payload = &data[span.payload.clone()];
        self.result = if recover {
            self.decoder
                .recover_block_into(payload, span.original_size, &mut self.output)
        } else {
            self.decoder
                .decode_block_into(payload, span.original_size, &mut self.output)
                .map(|()| BlockStatus::Intact)
        };
    }
}

/// Decode batches of blocks. Each slot reuses its decoder, so uneven
/// blocks cost no new state.
struct BatchDecoder {
    /// Block size of the frame.
    block_size: BlockSize,
    /// Thread count of the pool.
    workers: usize,
    /// True when a damaged block keeps its bytes.
    recover: bool,
    /// One slot for each block of the largest batch.
    slots: Vec<DecodeSlot>,
    /// Pool that the first batch of two blocks or more builds.
    pool: Option<rayon::ThreadPool>,
}

impl BatchDecoder {
    /// Start with no slots and no pool.
    const fn new(block_size: BlockSize, workers: usize, recover: bool) -> Self {
        Self {
            block_size,
            workers,
            recover,
            slots: Vec::new(),
            pool: None,
        }
    }

    /// Decode `spans` and return one slot for each span, in span order.
    ///
    /// One span runs on the caller thread. More spans run on the pool.
    /// A slot holds its own failure, so the caller can write the blocks
    /// before it.
    fn decode(&mut self, data: &[u8], spans: &[BlockSpan]) -> Result<&[DecodeSlot], Error> {
        use rayon::prelude::*;
        if spans.is_empty() {
            return Ok(&[]);
        }
        for _ in self.slots.len()..spans.len() {
            self.slots.push(DecodeSlot::new(self.block_size)?);
        }
        let slots = &mut self.slots[..spans.len()];
        let recover = self.recover;
        if let ([slot], [span]) = (&mut *slots, spans) {
            slot.decode(data, span, recover);
            return Ok(slots);
        }
        let pool = match self.pool.take() {
            Some(pool) => pool,
            None => rayon::ThreadPoolBuilder::new()
                .num_threads(self.workers)
                .build()
                .map_err(|_| Error::ThreadPoolFailed)?,
        };
        pool.install(|| {
            slots
                .par_iter_mut()
                .zip(spans.par_iter())
                .for_each(|(slot, span)| slot.decode(data, span, recover));
        });
        self.pool = Some(pool);
        Ok(slots)
    }
}

/// Print one line about a damaged block in recover mode.
///
/// `start` is the output offset of the block and `len` is its length.
fn report_damage(number: u64, offset: usize, fault: &Error, kept: bool, start: u64, len: usize) {
    let range = if len == 0 {
        String::new()
    } else {
        let last = start + len as u64 - 1;
        let state = if kept {
            "keep the decoded bytes"
        } else {
            "are zero"
        };
        format!(" Output bytes {start} to {last} {state}.")
    };
    eprintln!("Writing invalid block {number} at input offset {offset}: {fault}.{range}");
}

/// Run the decode and test paths.
///
/// Take up to one block for each worker from the block list, decode
/// them together, and write them in order. Early blocks reach the
/// output before a late failure, like the reference. A test run
/// writes to a sink. Memory stays near one block for each worker.
///
/// Recover mode writes every block. A damaged block prints a report.
/// The run ends with code 2 when it found one. A bad block header or a
/// short input stops the run after the earlier blocks.
fn decode(
    config: &Config,
    input: &mut dyn Read,
    output: &mut dyn Write,
    file: Option<&str>,
) -> Result<(), ExitError> {
    let data = read_all(input)?;
    let recover = config.mode == Mode::Recover;
    let block_size = parse_stream_header(&data, recover)?;
    let workers = config.workers;
    let codec_error = |error: &Error| ExitError::from_codec(error, Mode::Decode, workers > 1);
    let mut blocks = BlockSpans::new(&data, block_size);
    let mut batch_decoder = BatchDecoder::new(block_size, workers, recover);
    let mut spans = Vec::with_capacity(workers);
    let mut written: u64 = 0;
    let mut number: u64 = 0;
    let mut damaged: u64 = 0;
    loop {
        let mut fault = None;
        spans.clear();
        spans.extend(blocks.by_ref().take(workers).map_while(|item| match item {
            Ok(span) => Some(span),
            Err(error) => {
                fault = Some(error);
                None
            }
        }));
        let slots = batch_decoder
            .decode(&data, &spans)
            .map_err(|error| codec_error(&error))?;
        for (slot, span) in slots.iter().zip(&spans) {
            number += 1;
            let len = slot.output.len();
            match &slot.result {
                Ok(BlockStatus::Intact) => {}
                Ok(BlockStatus::BadChecksum) => {
                    report_damage(
                        number,
                        span.offset,
                        &Error::ChecksumMismatch,
                        true,
                        written,
                        len,
                    );
                    damaged += 1;
                }
                Ok(BlockStatus::Zeroed(error)) => {
                    report_damage(number, span.offset, error, false, written, len);
                    damaged += 1;
                }
                Err(error) => return Err(codec_error(error)),
            }
            output
                .write_all(&slot.output)
                .map_err(|error| codec_error(&Error::Io(error)))?;
            written = written.saturating_add(len as u64);
        }
        if let Some(error) = fault {
            if recover {
                // The caller keeps the output, so write the blocks before the fault.
                let _ = output.flush();
                eprintln!(
                    "Recovery stopped at block {}, input offset {}. Output bytes written: {written}. Damaged blocks: {damaged}.",
                    number + 1,
                    blocks.fault_at,
                );
            }
            return Err(error);
        }
        if spans.len() < workers {
            finish(output, config, file, data.len(), written)?;
            return if damaged == 0 {
                Ok(())
            } else {
                Err(ExitError::corrupt(format!(
                    "Recovery finished. Damaged blocks: {damaged}."
                )))
            };
        }
    }
}

/// Run one stream through the selected mode.
fn run_stream(
    config: &Config,
    input: &mut dyn Read,
    output: &mut dyn Write,
    file: Option<&str>,
) -> Result<(), ExitError> {
    match config.mode {
        Mode::Encode => encode(config, input, output, file),
        Mode::Decode | Mode::Test | Mode::Recover => decode(config, input, output, file),
    }
}

/// Append `suffix` to the whole file name.
fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Check for a `.bz3` suffix, case-sensitively like the reference.
fn has_bz3_suffix(path: &Path) -> bool {
    path.as_os_str().as_encoded_bytes().ends_with(b".bz3")
}

/// Strip `.bz3` from a decode input name, like the reference.
/// The match is case-sensitive. A name that is only `.bz3` stays.
fn strip_bz3_suffix(path: &Path) -> Option<PathBuf> {
    path.extension()
        .is_some_and(|extension| extension == "bz3")
        .then(|| path.with_extension(""))
}

/// Derive the output path for one input file.
///
/// Returns `None` for standard output: with `-c` and in test mode.
/// Encode appends `.bz3`. Decode strips it, or falls back to `.out`
/// with a notice unless `-q` suppresses the notice.
fn derive_output_path(path: &Path, name: &str, config: &Config) -> Option<PathBuf> {
    if config.to_stdout {
        return None;
    }
    match config.mode {
        Mode::Test => None,
        Mode::Encode => Some(append_suffix(path, ".bz3")),
        Mode::Decode | Mode::Recover => Some(strip_bz3_suffix(path).unwrap_or_else(|| {
            if !config.quiet {
                eprintln!("bzip3: Can't guess original name for {name} -- using {name}.out");
            }
            append_suffix(path, ".out")
        })),
    }
}

/// Copy mode and modification time from the input to the open output.
/// Best effort: all failures stay silent, like the reference.
fn copy_metadata(metadata: &Metadata, file: &File) {
    #[cfg(unix)]
    let _ = file.set_permissions(metadata.permissions());
    if let Ok(modified) = metadata.modified() {
        let _ = file.set_modified(modified);
    }
}

/// Remove a partial output after a failure. Errors stay silent.
fn remove_partial_output(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Run one positional file.
///
/// Order: suffix guard, open input, derive output, open output,
/// terminal check and run. Failures remove a created file output.
/// Success copies metadata and removes the input unless `-k`,
/// `-c`, test mode, or recover mode applies. Recover mode keeps its
/// output after a failure.
fn run_file(config: &Config, path: &Path) -> Result<(), ExitError> {
    let name = path.to_string_lossy();
    if config.mode == Mode::Encode && has_bz3_suffix(path) {
        return Err(ExitError::usage(format!(
            "bzip3: Input file {name} already has .bz3 suffix."
        )));
    }
    let (mut input, metadata) = open_input(path, &name)?;
    let Some(target) = derive_output_path(path, &name, config) else {
        if config.mode == Mode::Test {
            return run_stream(config, &mut input, &mut std::io::sink(), Some(&name));
        }
        check_terminal(config.mode, false, std::io::stdout().is_terminal())?;
        return run_stream(
            config,
            &mut input,
            &mut std::io::stdout().lock(),
            Some(&name),
        );
    };
    let mut output = open_output(&target, config.force)?;
    let recover = config.mode == Mode::Recover;
    if let Err(error) = run_stream(config, &mut input, &mut output, Some(&name)) {
        if recover {
            // Recover mode keeps the data that it wrote.
            copy_metadata(&metadata, &output);
        } else {
            remove_partial_output(&target);
        }
        return Err(error);
    }
    // Success keeps the input metadata on the output, like the reference.
    copy_metadata(&metadata, &output);
    if !config.keep && !recover {
        std::fs::remove_file(path).map_err(|error| {
            ExitError::usage(format!(
                "Error: failed to remove input file `{name}': {}",
                os_error_text(&error)
            ))
        })?;
    }
    Ok(())
}

/// Run the standard-input to standard-output flow.
fn run_stdin(config: &Config) -> Result<(), ExitError> {
    check_terminal(
        config.mode,
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
    )?;
    let mut input = std::io::stdin().lock();
    if config.mode == Mode::Test {
        run_stream(config, &mut input, &mut std::io::sink(), None)
    } else {
        run_stream(config, &mut input, &mut std::io::stdout().lock(), None)
    }
}

/// Print the error of a result. Returns the exit code.
fn report(result: Result<(), ExitError>) -> u8 {
    result.map_or_else(
        |error| {
            eprintln!("{error}");
            error.code
        },
        |()| 0,
    )
}

/// Run every input. Empty input runs the stdin flow; otherwise each
/// positional runs alone. The exit code is the maximum per-file code.
/// A failure to flush standard output at the end is fatal with code 1.
fn run(config: &Config) -> ExitCode {
    let code = if config.files.is_empty() {
        report(run_stdin(config))
    } else {
        config
            .files
            .iter()
            .map(|path| report(run_file(config, path)))
            .max()
            .unwrap_or(0)
    };
    if let Err(error) = std::io::stdout().flush() {
        eprintln!("Error: Failed on fclose(stdout): {}", os_error_text(&error));
        return ExitCode::FAILURE;
    }
    ExitCode::from(code)
}

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
        Err(Stop::Error(error)) => ExitCode::from(report(Err(error))),
    }
}

#[cfg(test)]
mod tests {
    use super::Alias;
    use super::BlockSizeArg;
    use super::Config;
    use super::MAX_WORKERS;
    use super::Mode;
    use super::Stop;
    use super::check_terminal;
    use clap::error::ErrorKind;

    use super::format_verbose_line;
    use super::infer_block_size;
    use super::parse_block_size;
    use super::parse_config;
    use super::parse_level_token;
    use super::resolve_workers;
    use super::rewrite_level_tokens;
    use super::strip_bz3_suffix;
    use bzip3_rs::BlockSize;
    use bzip3_rs::Jobs;
    use bzip3_rs::default_workers;
    use std::ffi::OsStr;
    use std::ffi::OsString;
    use std::path::Path;
    use std::path::PathBuf;

    #[test]
    fn infer_block_size_follows_block_headers() {
        let plain = vec![b'A'; 3 * BlockSize::MIN.as_usize() / 2];
        let coded = bzip3_rs::compress(&plain, BlockSize::MIN).unwrap();
        assert_eq!(infer_block_size(&coded), BlockSize::MIN);
        // A frame header with no blocks.
        assert_eq!(infer_block_size(&coded[..9]), BlockSize::MIN);
        // A block header with an original size above the minimum.
        let mut big = coded[..9].to_vec();
        big.extend_from_slice(&0_u32.to_le_bytes());
        big.extend_from_slice(&(1_u32 << 20).to_le_bytes());
        assert_eq!(infer_block_size(&big), BlockSize::new(1 << 20).unwrap());
        // A block header that claims more bytes than the input holds.
        let mut cut = coded[..9].to_vec();
        cut.extend_from_slice(&u32::MAX.to_le_bytes());
        cut.extend_from_slice(&(1_u32 << 20).to_le_bytes());
        assert_eq!(infer_block_size(&cut), BlockSize::MIN);
    }

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
    fn tty_refusal_matches_reference() {
        assert!(check_terminal(Mode::Encode, false, true).is_err());
        assert!(check_terminal(Mode::Encode, true, false).is_ok());
        assert!(check_terminal(Mode::Decode, true, false).is_err());
        assert!(check_terminal(Mode::Decode, false, true).is_ok());
        assert!(check_terminal(Mode::Test, true, false).is_err());
        assert!(check_terminal(Mode::Test, false, true).is_ok());
        assert!(check_terminal(Mode::Recover, true, false).is_err());
        assert!(check_terminal(Mode::Recover, false, true).is_ok());
    }

    #[test]
    fn verbose_lines_match_reference() {
        assert_eq!(
            format_verbose_line(Mode::Encode, Some("empty.bin"), 0, 9),
            " empty.bin:\t0 -> 9 bytes, inf%, inf bpb\n"
        );
        assert_eq!(
            format_verbose_line(Mode::Decode, Some("hello.bin.bz3"), 49, 24),
            " hello.bin.bz3:\t49 -> 24 bytes, 204.17%, 16.33 bpb\n"
        );
        assert_eq!(
            format_verbose_line(Mode::Test, Some("hello.bin.bz3"), 49, 24),
            " hello.bin.bz3:\tOK, 49 -> 24 bytes, 204.17%, 16.33 bpb\n"
        );
        assert_eq!(
            format_verbose_line(Mode::Encode, None, 1_000, 500),
            "\t1000 -> 500 bytes, 50.00%, 4.00 bpb\n"
        );
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
    fn suffix_rule_matches_reference() {
        let strip = |name: &str| strip_bz3_suffix(Path::new(name));
        assert_eq!(strip("a.bz3"), Some(PathBuf::from("a")));
        assert_eq!(strip("ab.bz3"), Some(PathBuf::from("ab")));
        assert_eq!(strip("dir/a.txt.bz3"), Some(PathBuf::from("dir/a.txt")));
        assert_eq!(strip(".bz3"), None);
        assert_eq!(strip("a.txt"), None);
        assert_eq!(strip("a.BZ3"), None);
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

    fn test_config(mode: Mode, to_stdout: bool, quiet: bool) -> Config {
        Config {
            mode,
            to_stdout,
            force: false,
            keep: true,
            verbose: false,
            quiet,
            block_size: BlockSize::MIN,
            workers: 1,
            files: Vec::new(),
        }
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

    #[test]
    fn parse_level_token_covers_edges() {
        use std::os::unix::ffi::OsStrExt as _;
        assert_eq!(parse_level_token(OsStr::new("-1")), Some(1));
        assert_eq!(parse_level_token(OsStr::new("-14")), Some(14));
        assert_eq!(parse_level_token(OsStr::new("-01")), Some(1));
        for text in ["", "-", "--1", "-0", "-15", "-19", "1", "-1a", "-x", "--"] {
            assert_eq!(parse_level_token(OsStr::new(text)), None, "{text}");
        }
        let raw = OsStr::from_bytes(b"\xff\xfe");
        assert_eq!(parse_level_token(raw), None);
        let raw = OsStr::from_bytes(b"-\xff");
        assert_eq!(parse_level_token(raw), None);
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

    #[test]
    fn derive_output_path_covers_modes() {
        use super::derive_output_path;
        let path = Path::new("plain.txt");
        assert!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Test, false, false)).is_none()
        );
        assert!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Encode, true, false))
                .is_none()
        );
        assert_eq!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Encode, false, false)),
            Some(PathBuf::from("plain.txt.bz3"))
        );
        assert_eq!(
            derive_output_path(
                Path::new("a.bz3"),
                "a.bz3",
                &test_config(Mode::Decode, false, false)
            ),
            Some(PathBuf::from("a"))
        );
        assert_eq!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Decode, false, true)),
            Some(PathBuf::from("plain.txt.out"))
        );
        assert_eq!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Decode, false, false)),
            Some(PathBuf::from("plain.txt.out"))
        );
        assert_eq!(
            derive_output_path(path, "plain.txt", &test_config(Mode::Recover, false, true)),
            Some(PathBuf::from("plain.txt.out"))
        );
    }

    #[test]
    fn block_spans_reports_all_faults() {
        use super::BlockSpans;
        use bzip3_rs::StreamHeader;
        let input = b"hello world ".repeat(200);
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let spans = BlockSpans::new(&coded, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(spans.len(), 1);
        let head_only = &coded[..StreamHeader::LEN];
        assert!(BlockSpans::new(head_only, BlockSize::MIN).next().is_none());
        let cut = &coded[..StreamHeader::LEN + 3];
        let mut walk = BlockSpans::new(cut, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        assert!(walk.next().is_none());
        let mut bad_comp = coded.clone();
        bad_comp[StreamHeader::LEN..StreamHeader::LEN + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut walk = BlockSpans::new(&bad_comp, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        let mut bad_orig = coded.clone();
        bad_orig[StreamHeader::LEN + 4..StreamHeader::LEN + 8]
            .copy_from_slice(&(BlockSize::MIN.get() + 1).to_le_bytes());
        let mut walk = BlockSpans::new(&bad_orig, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        let mut short = coded.clone();
        let keep = short.len() - 1;
        short.truncate(keep);
        let mut walk = BlockSpans::new(&short, BlockSize::MIN);
        assert!(matches!(walk.next(), Some(Err(_))));
        assert!(walk.next().is_none());
    }

    #[test]
    fn batch_decoder_covers_single_empty_and_reuse() {
        use super::BatchDecoder;
        use super::BlockSpans;
        use bzip3_rs::BlockStatus;
        let input = vec![b'q'; 3 * BlockSize::MIN.as_usize() + 100];
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        let spans = BlockSpans::new(&coded, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(spans.len() >= 3);
        let mut solo = BatchDecoder::new(BlockSize::MIN, 1, false);
        let empty = solo.decode(&coded, &[]).unwrap();
        assert_eq!(empty.len(), 0);
        let one = solo.decode(&coded, &spans[..1]).unwrap();
        assert_eq!(one.len(), 1);
        assert!(matches!(one[0].result, Ok(BlockStatus::Intact)));
        assert_eq!(one[0].output.len(), BlockSize::MIN.as_usize());
        let mut pool = BatchDecoder::new(BlockSize::MIN, 2, false);
        let first = pool.decode(&coded, &spans[..2]).unwrap().len();
        assert_eq!(first, 2);
        let second = pool.decode(&coded, &spans[2..]).unwrap().len();
        assert_eq!(second, spans.len() - 2);
        let mut damaged = coded.clone();
        damaged[spans[0].payload.start] ^= 0xFF;
        let hurt = BlockSpans::new(&damaged, BlockSize::MIN)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut plain = BatchDecoder::new(BlockSize::MIN, 1, false);
        let slots = plain.decode(&damaged, &hurt[..1]).unwrap();
        assert!(!matches!(
            slots[0].result,
            Ok(bzip3_rs::BlockStatus::Intact)
        ));
        let mut kept = BatchDecoder::new(BlockSize::MIN, 1, true);
        let slots = kept.decode(&damaged, &hurt[..1]).unwrap();
        assert!(!matches!(slots[0].result, Ok(BlockStatus::Intact)));
    }

    #[test]
    fn report_damage_handles_empty_range() {
        use super::report_damage;
        use bzip3_rs::error::Error;
        report_damage(1, 0, &Error::InvalidHeader, true, 0, 0);
        report_damage(2, 9, &Error::ChecksumMismatch, true, 0, 10);
        report_damage(3, 9, &Error::TruncatedData, false, 10, 5);
    }

    #[test]
    fn infer_block_size_clamps_small() {
        use bzip3_rs::StreamHeader;
        let mut data = StreamHeader::new(BlockSize::MIN).to_bytes().to_vec();
        data.extend_from_slice(&0_u32.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        assert_eq!(infer_block_size(&data), BlockSize::MIN);
    }

    #[test]
    fn stream_header_parse_covers_branches() {
        use super::parse_stream_header;
        assert!(parse_stream_header(b"NOPE!", false).is_err());
        assert!(parse_stream_header(b"BZ3v1", false).is_err());
        let mut bad = b"BZ3v1".to_vec();
        bad.extend_from_slice(&0_u32.to_le_bytes());
        assert!(parse_stream_header(&bad, false).is_err());
        let input = b"hello world ".repeat(50);
        let mut coded = Vec::new();
        bzip3_rs::stream::compress_slice(&input, &mut coded, BlockSize::MIN, Jobs::SERIAL).unwrap();
        assert_eq!(parse_stream_header(&coded, false).unwrap(), BlockSize::MIN);
        let mut broken = coded.clone();
        broken[5..9].copy_from_slice(&0_u32.to_le_bytes());
        assert!(parse_stream_header(&broken, true).is_ok());
    }

    struct FailIo(&'static str);

    impl std::io::Read for FailIo {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, self.0))
        }
    }

    impl std::io::Write for FailIo {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, self.0))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, self.0))
        }
    }

    #[test]
    fn file_helpers_report_io_faults() {
        use super::finish;
        use super::open_input;
        use super::open_output;
        use super::read_all;
        let dir = std::env::temp_dir().join(format!("bzip3-main-{}-helpers", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("gone.bin");
        assert!(open_input(&missing, "gone.bin").is_err());
        assert!(open_input(&dir, "dir").is_err());
        assert!(open_output(&dir, false).is_err());
        let taken = dir.join("taken.bin");
        std::fs::write(&taken, b"x").unwrap();
        assert!(open_output(&taken, false).is_err());
        assert!(open_output(&taken, true).is_ok());
        let nowhere = dir.join("nope").join("out.bin");
        assert!(open_output(&nowhere, false).is_err());
        let mut fail = FailIo("pipe broke");
        assert!(read_all(&mut fail).is_err());
        let config = test_config(Mode::Encode, false, false);
        assert!(finish(&mut fail, &config, None, 1, 9).is_err());
        {
            assert!((&mut fail as &mut dyn std::io::Write).write(b"x").is_err());
        }
        let mut sink = Vec::new();
        assert!(finish(&mut sink, &config, None, 1, 9).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
