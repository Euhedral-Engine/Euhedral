//! The log: one line per record on the standard output, `LEVEL message`, at info and above.

#![allow(clippy::print_stdout, reason = "the log is the process's standard output")]

use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError, set_logger, set_max_level};

struct Stdout;

impl Log for Stdout {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            println!("{:>5} {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: Stdout = Stdout;

/// Installs the logger, once per process.
pub fn install_logger() -> Result<(), SetLoggerError> {
    set_logger(&LOGGER)?;
    set_max_level(LevelFilter::Info);
    Ok(())
}
