//! `quantization`: the quantization command line. See the library for the commands.

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

use euhedral_inference_quantize::run;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let outcome = run(&args);
    if !outcome.message.is_empty() {
        // A failed write to standard error has nowhere to be reported.
        let _ = io::stderr().write_all(outcome.message.as_bytes());
    }
    ExitCode::from(outcome.code)
}
