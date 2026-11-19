//! The time of every HTTP request by route template, method and status: from the request reaching the router to the last
//! byte handed to the connection. (The server hands the router a request whose head and body are both read, and gives it
//! no earlier time, so the read itself is not in the time.) Like the other series it is recorded with atomics in a table laid out when the registry is made, so
//! the labels are bounded by construction: a path that is no route is one label value, whatever its text.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use euhedral_net::Method;

use super::Metrics;
use super::window::WindowMax;

/// The route a request is counted under, the `uri` label: the route's template, never the path as it came.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uri {
    Health,
    Metrics,
    Models,
    Model,
    ChatCompletions,
    Messages,
    CountTokens,
    Responses,
    /// A path that is none of the routes.
    NotFound,
}

impl Uri {
    pub const ALL: [Uri; 9] = [
        Uri::Health,
        Uri::Metrics,
        Uri::Models,
        Uri::Model,
        Uri::ChatCompletions,
        Uri::Messages,
        Uri::CountTokens,
        Uri::Responses,
        Uri::NotFound,
    ];

    /// The label value.
    pub fn label(self) -> &'static str {
        match self {
            Uri::Health => "/health",
            Uri::Metrics => "/metrics",
            Uri::Models => "/v1/models",
            Uri::Model => "/v1/models/{model}",
            Uri::ChatCompletions => "/v1/chat/completions",
            Uri::Messages => "/v1/messages",
            Uri::CountTokens => "/v1/messages/count_tokens",
            Uri::Responses => "/v1/responses",
            Uri::NotFound => "/**",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The methods the table tells apart; a method that is none of them is counted under `OTHER`.
const METHODS: [&str; 10] = [
    "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "OTHER",
];

fn method_index(method: &Method) -> usize {
    match method {
        Method::Get => 0,
        Method::Head => 1,
        Method::Post => 2,
        Method::Put => 3,
        Method::Delete => 4,
        Method::Connect => 5,
        Method::Options => 6,
        Method::Trace => 7,
        Method::Patch => 8,
        Method::Other(_) => 9,
    }
}

/// The statuses the table tells apart: `200` and every client and server error code. The server answers with nothing
/// else; a status outside them is not recorded rather than counted under another code.
const FIRST_ERROR: u16 = 400;
const LAST_ERROR: u16 = 599;
const STATUS_COUNT: usize = 1 + (LAST_ERROR - FIRST_ERROR + 1) as usize;

fn status_index(status: u16) -> Option<usize> {
    match status {
        200 => Some(0),
        FIRST_ERROR..=LAST_ERROR => Some(1 + usize::from(status - FIRST_ERROR)),
        _ => None,
    }
}

fn status_of(index: usize) -> u16 {
    if index == 0 {
        200
    } else {
        FIRST_ERROR + (index - 1) as u16
    }
}

/// The class of a status, the `outcome` label.
pub fn outcome(status: u16) -> &'static str {
    match status / 100 {
        1 => "INFORMATIONAL",
        2 => "SUCCESS",
        3 => "REDIRECTION",
        4 => "CLIENT_ERROR",
        5 => "SERVER_ERROR",
        _ => "UNKNOWN",
    }
}

/// One label set's count, total time and largest time.
#[derive(Debug)]
pub(super) struct Cell {
    pub(super) count: AtomicU64,
    pub(super) sum_ns: AtomicU64,
    pub(super) max_ns: WindowMax,
}

/// The table of cells, one per route, method and status.
#[derive(Debug)]
pub(super) struct Table {
    cells: Box<[Cell]>,
}

impl Table {
    pub(super) fn new(now_ms: i64) -> Table {
        let size = Uri::ALL.len() * METHODS.len() * STATUS_COUNT;
        Table {
            cells: (0..size)
                .map(|_| Cell {
                    count: AtomicU64::new(0),
                    sum_ns: AtomicU64::new(0),
                    max_ns: WindowMax::new(now_ms),
                })
                .collect(),
        }
    }

    fn cell(&self, uri: Uri, method: usize, status: usize) -> &Cell {
        &self.cells[(uri.index() * METHODS.len() + method) * STATUS_COUNT + status]
    }

    fn record(&self, uri: Uri, method: &Method, status: u16, nanos: u64, now_ms: i64) {
        let Some(status) = status_index(status) else {
            return;
        };
        let cell = self.cell(uri, method_index(method), status);
        cell.count.fetch_add(1, Ordering::Relaxed);
        cell.sum_ns.fetch_add(nanos, Ordering::Relaxed);
        cell.max_ns.record(nanos, now_ms);
    }

    /// Every cell that saw a request, with its labels: route, method, status.
    pub(super) fn seen(&self) -> impl Iterator<Item = (Uri, &'static str, u16, &Cell)> {
        Uri::ALL.into_iter().flat_map(move |uri| {
            (0..METHODS.len()).flat_map(move |method| {
                (0..STATUS_COUNT).filter_map(move |status| {
                    let cell = self.cell(uri, method, status);
                    (cell.count.load(Ordering::Relaxed) > 0).then_some((uri, METHODS[method], status_of(status), cell))
                })
            })
        })
    }
}

/// The clock of one request. It is finished once, by whoever hands the last byte to the connection; a request that is
/// dropped unfinished before any head went out (the client left before an answer) is not recorded; one whose stream
/// was cut after its head is recorded when it is dropped.
#[derive(Debug)]
pub struct RequestTimer {
    metrics: Arc<Metrics>,
    uri: Uri,
    method: Method,
    started: Instant,
    finished: AtomicBool,
    /// The head of a streamed reply went out: the status is `200` whatever follows.
    opened: AtomicBool,
}

impl RequestTimer {
    /// Starts the clock of a request that was read now.
    pub fn start(metrics: Arc<Metrics>, uri: Uri, method: Method) -> RequestTimer {
        RequestTimer {
            metrics,
            uri,
            method,
            started: Instant::now(),
            finished: AtomicBool::new(false),
            opened: AtomicBool::new(false),
        }
    }

    /// The head of a streamed reply went out. From here the request is recorded as a `200` even if the stream is cut
    /// before its end.
    pub fn opened(&self) {
        self.opened.store(true, Ordering::Release);
    }

    /// Records the request as answered with `status`; a second call does nothing.
    pub fn finish(&self, status: u16) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let nanos = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.metrics
            .http
            .record(self.uri, &self.method, status, nanos, Metrics::wall_clock_ms());
    }
}

impl Drop for RequestTimer {
    /// A stream that was cut after its head (the client left, a write failed, the reader was too slow) is timed to the
    /// moment it was let go.
    fn drop(&mut self) {
        if self.opened.load(Ordering::Acquire) {
            self.finish(200);
        }
    }
}
