//! Throwaway one-shot encoder for the corpus oracle matrix.
//!
//! Built in the work dir (outside the repo) with a path dependency on
//! the crate under test. Encodes one file at one block size.

use std::process::ExitCode;

use bzip3_rs::BlockSize;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: encoder <block-size-bytes> <input> <output>");
        return ExitCode::from(2);
    }
    let size: u32 = match args[1].parse() {
        Ok(size) => size,
        Err(_) => {
            eprintln!("bad block size");
            return ExitCode::from(2);
        }
    };
    let block_size = match BlockSize::new(size) {
        Ok(block_size) => block_size,
        Err(_) => {
            eprintln!("block size outside 65 KiB..=511 MiB");
            return ExitCode::from(2);
        }
    };
    let input = match std::fs::read(&args[2]) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("read input: {err}");
            return ExitCode::from(1);
        }
    };
    let encoded = match bzip3_rs::compress(&input, block_size) {
        Ok(encoded) => encoded,
        Err(err) => {
            eprintln!("compress: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = std::fs::write(&args[3], &encoded) {
        eprintln!("write output: {err}");
        return ExitCode::from(1);
    }
    println!(
        "{}: {} -> {} bytes",
        args[2],
        input.len(),
        encoded.len()
    );
    ExitCode::SUCCESS
}
