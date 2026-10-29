//! File versus standard-stream execution.
//!
//! This module runs the encode, decode, test, and recover modes over
//! files or standard streams. It owns the output naming rules, the
//! metadata copy, and the cleanup rules. It depends on
//! [`crate::cli_args`] for the validated config and on
//! [`crate::cli_iter`] for block iteration and batch decode.

// Visibility stays `pub(crate)` for the binary entry.
// Reason: the parent module is private, so the
// nursery lint asks for plain `pub`, but `pub(crate)` states the
// crate-wide intent at each item.
#![expect(clippy::redundant_pub_crate)]

use std::fs::File;
use std::fs::Metadata;
use std::io::IsTerminal;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use bzip3_rs::BlockStatus;
use bzip3_rs::Jobs;
use bzip3_rs::error::Error;
use bzip3_rs::stream;

use crate::cli_args::Config;
use crate::cli_args::ExitError;
use crate::cli_args::Mode;
use crate::cli_args::os_error_text;
use crate::cli_iter::BatchDecoder;
use crate::cli_iter::BlockSpan;
use crate::cli_iter::BlockSpans;
use crate::cli_iter::DecodeSlot;
use crate::cli_iter::parse_stream_header;
use crate::cli_iter::report_damage;

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

/// Map an output open failure to its message.
fn output_open_error(path: &Path, error: &std::io::Error) -> ExitError {
    let name = path.display();
    ExitError::usage(if path.is_dir() {
        format!("Error: output file `{name}' is a directory.")
    } else if error.kind() == std::io::ErrorKind::AlreadyExists {
        format!("Error: output file `{name}' already exists. Use -f to force overwrite.")
    } else {
        format!(
            "Error: failed to open output file `{name}': {}",
            os_error_text(error)
        )
    })
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
    options
        .open(path)
        .map_err(|error| output_open_error(path, &error))
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

/// Progress of one decode run across batches.
#[derive(Default)]
struct DecodeProgress {
    /// Plain bytes written so far.
    written: u64,
    /// Blocks written so far.
    number: u64,
    /// Damaged blocks written so far.
    damaged: u64,
}

/// Fill `spans` with up to `workers` block spans.
///
/// A bad block header stops the batch and returns the fault. Earlier
/// spans of the batch stay for the caller to write first.
fn fill_batch(
    blocks: &mut BlockSpans<'_>,
    spans: &mut Vec<BlockSpan>,
    workers: usize,
) -> Option<ExitError> {
    let mut fault = None;
    spans.clear();
    spans.extend(blocks.by_ref().take(workers).map_while(|item| match item {
        Ok(span) => Some(span),
        Err(error) => {
            fault = Some(error);
            None
        }
    }));
    fault
}

/// Write one decoded slot and record its damage report.
fn write_slot(
    output: &mut dyn Write,
    slot: &DecodeSlot,
    span: &BlockSpan,
    progress: &mut DecodeProgress,
    parallel: bool,
) -> Result<(), ExitError> {
    progress.number += 1;
    let len = slot.output.len();
    match &slot.result {
        Ok(BlockStatus::Intact) => {}
        Ok(BlockStatus::BadChecksum) => {
            report_damage(
                progress.number,
                span.offset,
                &Error::ChecksumMismatch,
                true,
                progress.written,
                len,
            );
            progress.damaged += 1;
        }
        Ok(BlockStatus::Zeroed(error)) => {
            report_damage(
                progress.number,
                span.offset,
                error,
                false,
                progress.written,
                len,
            );
            progress.damaged += 1;
        }
        Err(error) => return Err(ExitError::from_codec(error, Mode::Decode, parallel)),
    }
    output
        .write_all(&slot.output)
        .map_err(|error| ExitError::from_codec(&Error::Io(error), Mode::Decode, parallel))?;
    progress.written = progress.written.saturating_add(len as u64);
    Ok(())
}

/// Write one batch of decoded slots in span order.
fn write_batch(
    output: &mut dyn Write,
    slots: &[DecodeSlot],
    spans: &[BlockSpan],
    progress: &mut DecodeProgress,
    parallel: bool,
) -> Result<(), ExitError> {
    for (slot, span) in slots.iter().zip(spans) {
        write_slot(output, slot, span, progress, parallel)?;
    }
    Ok(())
}

/// End a decode run after the last batch: flush, print the verbose
/// line, and report damaged blocks of recover mode.
fn finish_decode(
    config: &Config,
    output: &mut dyn Write,
    file: Option<&str>,
    input_len: usize,
    progress: &DecodeProgress,
) -> Result<(), ExitError> {
    finish(output, config, file, input_len, progress.written)?;
    if progress.damaged == 0 {
        Ok(())
    } else {
        Err(ExitError::corrupt(format!(
            "Recovery finished. Damaged blocks: {}.",
            progress.damaged
        )))
    }
}

/// Report a recover stop at a bad block header. The caller keeps the
/// output, so earlier blocks of the batch are already written.
fn report_recovery_stop(fault_at: usize, progress: &DecodeProgress, output: &mut dyn Write) {
    let _ = output.flush();
    eprintln!(
        "Recovery stopped at block {}, input offset {}. Output bytes written: {}. Damaged blocks: {}.",
        progress.number + 1,
        fault_at,
        progress.written,
        progress.damaged,
    );
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
    let parallel = config.workers > 1;
    let codec_error = |error: &Error| ExitError::from_codec(error, Mode::Decode, parallel);
    let mut blocks = BlockSpans::new(&data, block_size);
    let mut batch_decoder = BatchDecoder::new(block_size, config.workers, recover);
    let mut spans = Vec::with_capacity(config.workers);
    let mut progress = DecodeProgress::default();
    loop {
        let fault = fill_batch(&mut blocks, &mut spans, config.workers);
        let slots = batch_decoder
            .decode(&data, &spans)
            .map_err(|error| codec_error(&error))?;
        write_batch(output, slots, &spans, &mut progress, parallel)?;
        if let Some(error) = fault {
            if recover {
                report_recovery_stop(blocks.fault_at, &progress, output);
            }
            return Err(error);
        }
        if spans.len() < config.workers {
            return finish_decode(config, output, file, data.len(), &progress);
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

/// Fall back to `.out` when a decode input has no `.bz3` suffix.
/// Prints a notice unless `-q` suppresses it.
fn fallback_decode_output(path: &Path, name: &str, config: &Config) -> PathBuf {
    if !config.quiet {
        eprintln!("bzip3: Can't guess original name for {name} -- using {name}.out");
    }
    append_suffix(path, ".out")
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
        Mode::Decode | Mode::Recover => Some(
            strip_bz3_suffix(path).unwrap_or_else(|| fallback_decode_output(path, name, config)),
        ),
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

/// Reject an encode input that already has a `.bz3` suffix.
fn guard_encode_suffix(path: &Path, name: &str, config: &Config) -> Result<(), ExitError> {
    if config.mode == Mode::Encode && has_bz3_suffix(path) {
        return Err(ExitError::usage(format!(
            "bzip3: Input file {name} already has .bz3 suffix."
        )));
    }
    Ok(())
}

/// Run one positional file without a file output: test mode writes to
/// a sink, otherwise the data goes to standard output.
fn run_file_special(config: &Config, input: &mut File, name: &str) -> Result<(), ExitError> {
    if config.mode == Mode::Test {
        return run_stream(config, input, &mut std::io::sink(), Some(name));
    }
    check_terminal(config.mode, false, std::io::stdout().is_terminal())?;
    run_stream(config, input, &mut std::io::stdout().lock(), Some(name))
}

/// Run one positional file into `target`.
///
/// A failure removes a created file output, except in recover mode,
/// which keeps the data that it wrote. Success copies metadata and
/// removes the input unless `-k` or recover mode applies.
fn run_file_output(
    config: &Config,
    path: &Path,
    name: &str,
    input: &mut File,
    metadata: &Metadata,
    target: &Path,
    output: &mut File,
) -> Result<(), ExitError> {
    let recover = config.mode == Mode::Recover;
    if let Err(error) = run_stream(config, input, output, Some(name)) {
        if recover {
            // Recover mode keeps the data that it wrote.
            copy_metadata(metadata, output);
        } else {
            remove_partial_output(target);
        }
        return Err(error);
    }
    // Success keeps the input metadata on the output, like the reference.
    copy_metadata(metadata, output);
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

/// Run one positional file.
///
/// Order: suffix guard, open input, derive output, open output,
/// terminal check and run. Failures remove a created file output.
/// Success copies metadata and removes the input unless `-k`,
/// `-c`, test mode, or recover mode applies. Recover mode keeps its
/// output after a failure.
fn run_file(config: &Config, path: &Path) -> Result<(), ExitError> {
    let name = path.to_string_lossy();
    guard_encode_suffix(path, &name, config)?;
    let (mut input, metadata) = open_input(path, &name)?;
    let Some(target) = derive_output_path(path, &name, config) else {
        return run_file_special(config, &mut input, &name);
    };
    let mut output = open_output(&target, config.force)?;
    run_file_output(
        config,
        path,
        &name,
        &mut input,
        &metadata,
        &target,
        &mut output,
    )
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
pub(crate) fn run(config: &Config) -> ExitCode {
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

#[cfg(test)]
mod tests {
    use super::check_terminal;
    use super::derive_output_path;
    use super::finish;
    use super::format_verbose_line;
    use super::open_input;
    use super::open_output;
    use super::read_all;
    use crate::cli_args::Config;
    use crate::cli_args::Mode;
    use bzip3_rs::BlockSize;
    use std::path::Path;
    use std::path::PathBuf;

    use super::append_suffix;
    use super::has_bz3_suffix;
    use super::strip_bz3_suffix;

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
    fn suffix_rule_keeps_non_utf8_bytes() {
        // Build a name without valid UTF-8.
        // The file system stays out of this test.
        #[cfg(unix)]
        fn plain() -> PathBuf {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt as _;
            PathBuf::from(OsStr::from_bytes(b"\xff.bin"))
        }
        #[cfg(windows)]
        fn plain() -> PathBuf {
            use std::ffi::OsString;
            use std::os::windows::ffi::OsStringExt as _;
            let mut name = OsString::from_wide(&[0xD800]);
            name.push(".bin");
            PathBuf::from(name)
        }
        #[cfg(not(any(unix, windows)))]
        fn plain() -> PathBuf {
            PathBuf::from("skip.bin")
        }
        let path = plain();
        assert!(!has_bz3_suffix(&path));
        let coded = append_suffix(&path, ".bz3");
        assert!(has_bz3_suffix(&coded));
        assert_eq!(strip_bz3_suffix(&coded), Some(path));
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
    fn derive_output_path_covers_modes() {
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
