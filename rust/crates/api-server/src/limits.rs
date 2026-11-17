//! The limits of the server: what a request may carry and how long a connection may take. Every field is a setting;
//! none is tied to a machine.

use std::time::Duration;

use euhedral_inference_api_protocol::ApiError;
use euhedral_net::ServerConfig;

/// Limits on requests and connections.
///
/// The transport answers some requests by itself, before the handler runs, and its answers have an empty body and no
/// content type: 400 (a malformed request), 408 (a request that took longer than `read_limit` after its first byte),
/// 411 (a chunked request body), 413 for a body declared longer than `max_body_bytes + over_limit_read_bytes`, 414 (a
/// request line too long), 431 (header lines over `max_header_bytes`) and 505 (an unsupported HTTP version). Everything
/// else reaches the handler, which answers with the API's JSON error.
///
/// That is why the transport's body limit is `max_body_bytes + over_limit_read_bytes` and not `max_body_bytes`: the
/// transport refuses a declared body over its limit at the head, so a body between the two limits is read in full and
/// answered by the handler with the JSON 413. The price is that a connection may have the server buffer up to
/// `max_body_bytes + over_limit_read_bytes` bytes of a request that is then refused, and a transport answer closes the
/// connection after the server has read and discarded what the peer still sends, up to the size limits, for at most
/// the transport's linger time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The longest body of a generation request, in bytes. Longer is answered 413 with the API's error.
    pub max_body_bytes: usize,
    /// How many bytes beyond `max_body_bytes` the transport still reads so that the API can answer the JSON 413.
    pub over_limit_read_bytes: usize,
    /// The most connections open at once.
    pub max_connections: usize,
    /// The most bytes of a request's header lines; more is the transport's empty 431. The default, 8 KiB, is the limit of
    /// common HTTP servers.
    pub max_header_bytes: usize,
    /// How long a whole request may take to arrive from its first byte (the transport answers 408 after it).
    pub read_limit: Option<Duration>,
    /// How long a connection may wait for the first byte of a request; it is closed without an answer after it.
    pub idle_limit: Option<Duration>,
    /// How long one reply may take to go out.
    pub write_limit: Option<Duration>,
    /// How many generation requests may wait for a free slot; a request that finds the queue full is refused. Zero
    /// refuses every request that cannot start at once.
    pub max_queued: usize,
    /// How long a request may wait for a slot before it is refused with the timeout error. `None`: as long as the
    /// request limit allows.
    pub queue_limit: Option<Duration>,
    /// How long a request may take from its arrival, waiting and generating together; after it a waiting request is
    /// refused and a running one is interrupted, both with the timeout error.
    pub request_limit: Duration,
    /// How long a stop waits for the running generations to end before it reports itself done anyway.
    pub shutdown_limit: Duration,
    /// How long a stop lets the requests in flight (the waiting ones included) finish before admission refuses the
    /// waiting ones and interrupts the running ones.
    pub drain_limit: Duration,
    /// The most text a streamed reply takes in, counted in bytes of what the model said (not of the events that
    /// repeat it), while one piece is being written or while the owner has not yet taken it. A client that does not
    /// read lets it pile up; past this the generation is cancelled and the stream is cut. It does not apply once the
    /// end of the stream is queued.
    pub max_held_bytes: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        let transport = ServerConfig::default();
        Limits {
            max_body_bytes: 1024 * 1024,
            over_limit_read_bytes: 1024 * 1024,
            max_connections: transport.max_connections,
            max_header_bytes: 8 * 1024,
            read_limit: transport.read_limit,
            idle_limit: transport.idle_limit,
            write_limit: transport.write_limit,
            max_queued: 16,
            queue_limit: None,
            request_limit: Duration::from_secs(30 * 60),
            shutdown_limit: Duration::from_secs(30),
            drain_limit: Duration::from_secs(30),
            max_held_bytes: 1024 * 1024,
        }
    }
}

impl Limits {
    /// The transport's configuration for these limits.
    pub(crate) fn transport(&self) -> ServerConfig {
        ServerConfig {
            max_connections: self.max_connections,
            max_header_bytes: self.max_header_bytes,
            max_body_bytes: self.max_body_bytes.saturating_add(self.over_limit_read_bytes),
            read_limit: self.read_limit,
            idle_limit: self.idle_limit,
            write_limit: self.write_limit,
            ..ServerConfig::default()
        }
    }

    /// Whether a generation request body of `length` bytes is within the limit.
    pub fn check_body(&self, length: usize) -> Result<(), ApiError> {
        if length > self.max_body_bytes {
            Err(ApiError::request_too_large())
        } else {
            Ok(())
        }
    }
}
