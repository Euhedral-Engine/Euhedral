//! The time of every HTTP request by route template, method and status: from the head read to the last byte handed to
//! the connection. Like the other series it is recorded with atomics in a table laid out when the registry is made, so
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
            Uri::NotFound => "NOT_FOUND",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The methods the table tells apart; any other is counted under `Other`.
const METHODS: [&str; 5] = ["GET", "HEAD", "OPTIONS", "POST", "OTHER"];

fn method_index(method: &Method) -> usize {
    match method {
        Method::Get => 0,
        Method::Head => 1,
        Method::Options => 2,
        Method::Post => 3,
        _ => 4,
    }
}

/// The statuses the table tells apart: every one the server answers with. Any other is counted under the first of its
/// class, so the table stays small and the classes (the `outcome` label) stay right.
const STATUSES: [u16; 15] = [
    200, 400, 401, 403, 404, 405, 406, 408, 413, 415, 422, 429, 500, 503, 529,
];

fn status_index(status: u16) -> usize {
    if let Some(at) = STATUSES.iter().position(|each| *each == status) {
        return at;
    }
    let class = status / 100;
    STATUSES.iter().position(|each| each / 100 == class).unwrap_or(0)
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
        let size = Uri::ALL.len() * METHODS.len() * STATUSES.len();
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
        &self.cells[(uri.index() * METHODS.len() + method) * STATUSES.len() + status]
    }

    fn record(&self, uri: Uri, method: &Method, status: u16, nanos: u64, now_ms: i64) {
        let cell = self.cell(uri, method_index(method), status_index(status));
        cell.count.fetch_add(1, Ordering::Relaxed);
        cell.sum_ns.fetch_add(nanos, Ordering::Relaxed);
        cell.max_ns.record(nanos, now_ms);
    }

    /// Every cell that saw a request, with its labels: route, method, status.
    pub(super) fn seen(&self) -> impl Iterator<Item = (Uri, &'static str, u16, &Cell)> {
        Uri::ALL.into_iter().flat_map(move |uri| {
            (0..METHODS.len()).flat_map(move |method| {
                (0..STATUSES.len()).filter_map(move |status| {
                    let cell = self.cell(uri, method, status);
                    (cell.count.load(Ordering::Relaxed) > 0).then_some((uri, METHODS[method], STATUSES[status], cell))
                })
            })
        })
    }
}

/// The clock of one request. It is finished once, by whoever hands the last byte to the connection; a request that is
/// dropped unfinished (the client left before an answer) is not recorded.
#[derive(Debug)]
pub struct RequestTimer {
    metrics: Arc<Metrics>,
    uri: Uri,
    method: Method,
    started: Instant,
    finished: AtomicBool,
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
        }
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
