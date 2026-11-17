//! The handlers of [`signals::install`], in a process of their own: a signal that arrives before anyone waits for it
//! is kept, and the wait returns its exit code.
#![cfg(unix)]
#![allow(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread waits for the signal it sent itself"
)]

use std::process::{Command, id};

use euhedral_inference_server::EXIT_SIGTERM;
use euhedral_inference_server::signals::install;

#[test]
fn a_signal_sent_before_the_wait_is_kept() {
    let mut signals = install().expect("the handlers");
    let sent = Command::new("kill")
        .arg("-TERM")
        .arg(id().to_string())
        .status()
        .expect("kill");
    assert!(sent.success());
    assert_eq!(signals.wait(), EXIT_SIGTERM);
}
