//! Shared helpers for CLI integration tests.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

static NEXT: AtomicU64 = AtomicU64::new(0);

/// Scratch directory for one test. Removed on drop.
pub struct Scratch {
    /// Path of the scratch directory.
    pub dir: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Make an empty scratch directory for `case`.
#[must_use]
pub fn scratch(case: &str) -> Scratch {
    let id = NEXT.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("bzip3cli-{}-{id}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make scratch dir");
    Scratch { dir }
}

/// Path of the built `bzip3` binary.
#[must_use]
pub fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bzip3"))
}

/// Write `bytes` to `name` inside `dir`. Return the full path.
pub fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write scratch file");
    path
}

/// Run the binary in `dir` with `args` and `stdin`. Return the output.
pub fn run(dir: &Path, args: &[&str], stdin: &[u8]) -> Output {
    use std::process::Stdio;
    let mut child = Command::new(bin())
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bzip3");
    // The program can exit before it reads stdin, for example on a bad
    // flag. A broken pipe then is not a test fault.
    let written = child.stdin.as_mut().expect("open stdin").write_all(stdin);
    if let Err(fault) = written {
        assert_eq!(fault.kind(), std::io::ErrorKind::BrokenPipe, "write stdin");
    }
    child.wait_with_output().expect("wait bzip3")
}

/// Build `len` deterministic bytes from `seed` (xorshift64).
#[must_use]
pub fn pseudo(seed: u64, len: usize) -> Vec<u8> {
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state >> 33) as u8);
    }
    out
}

use std::io::Write as _;
