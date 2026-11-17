//! Process CPU time of an idle server. Ignored: it takes a minute. Run it with
//! `cargo test -p euhedral-inference-api-server --test idle -- --ignored --nocapture`.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::print_stderr,
    reason = "measurement: the test thread reads /proc, holds a socket and sleeps for the measured time"
)]

mod common;

use std::fs::read_to_string;
use std::net::TcpStream;
use std::thread::{available_parallelism, sleep};
use std::time::Duration;

use common::{Fixture, small_limits};
use euhedral_inference_api_server::Limits;

/// How long each case stays idle.
const IDLE: Duration = Duration::from_secs(30);

/// User plus system CPU seconds of this process, from `/proc/self/stat` (fields 14 and 15, in clock ticks of
/// `USER_HZ`, which is 100 on every Linux).
fn cpu_seconds() -> f64 {
    let stat = read_to_string("/proc/self/stat").expect("/proc/self/stat");
    // The command name may hold spaces and is in parentheses: the fields after it start after the last `)`.
    let after = &stat[stat.rfind(')').expect("a command name") + 2..];
    let fields: Vec<&str> = after.split(' ').collect();
    // `after` starts at field 3 (state), so fields 14 and 15 are at 11 and 12.
    let ticks = |index: usize| fields[index].parse::<f64>().expect("a number");
    (ticks(11) + ticks(12)) / 100.0
}

fn measure(connected: bool) -> (f64, usize) {
    let limits = Limits {
        idle_limit: None,
        read_limit: None,
        ..small_limits()
    };
    // The lattice has its default worker count.
    let fixture = Fixture::with(limits, None);
    // The server is up and has answered once, so that its start is not part of the idle time.
    assert_eq!(fixture.get("/health").status, 200);
    let connection = connected.then(|| TcpStream::connect(fixture.addr).expect("a connection"));
    let before = cpu_seconds();
    sleep(IDLE);
    let used = cpu_seconds() - before;
    drop(connection);
    let workers = available_parallelism().map_or(0, usize::from);
    fixture.finish();
    (used, workers)
}

#[test]
#[ignore = "measures 30 s of idleness twice"]
fn idle_cpu() {
    let (none, workers) = measure(false);
    eprintln!(
        "idle {} s, no connection: {none:.2} CPU s (available parallelism {workers})",
        IDLE.as_secs()
    );
    let (one, _) = measure(true);
    eprintln!(
        "idle {} s, one idle keep-alive connection: {one:.2} CPU s",
        IDLE.as_secs()
    );
}
