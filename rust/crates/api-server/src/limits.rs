//! The limits of the server: what a request may carry and how long a connection may take. Every field is a setting;
//! none is tied to a machine.

use std::time::Duration;

use euhedral_inference_api_protocol::ApiError;
use euhedral_net::ServerConfig;

/// Limits on requests and connections.
///
/// `euhedral-net` refuses a body longer than its own limit with an empty 413 before the handler runs, so a request
/// over `max_body_bytes` could not be answered with the API's JSON error. The transport's limit is therefore
/// `max_body_bytes + over_limit_read_bytes`: a body between the two is read and answered with the JSON error, and only a
/// declared length beyond both gets the transport's empty 413.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The longest body of a generation request, in bytes. Longer is answered 413 with the API's error.
    pub max_body_bytes: usize,
    /// How many bytes beyond `max_body_bytes` the transport still reads so that the API can answer the JSON 413.
    pub over_limit_read_bytes: usize,
    /// The most connections open at once.
    pub max_connections: usize,
    /// The most bytes of a request's header lines (more is the transport's empty 431).
    pub max_header_bytes: usize,
    /// How long a whole request may take to arrive from its first byte (the transport answers 408 after it).
    pub read_limit: Option<Duration>,
    /// How long a connection may wait for the first byte of a request; it is closed without an answer after it.
    pub idle_limit: Option<Duration>,
    /// How long one reply may take to go out.
    pub write_limit: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Limits {
        let transport = ServerConfig::default();
        Limits {
            max_body_bytes: 1024 * 1024,
            over_limit_read_bytes: 1024 * 1024,
            max_connections: transport.max_connections,
            max_header_bytes: transport.max_header_bytes,
            read_limit: transport.read_limit,
            idle_limit: transport.idle_limit,
            write_limit: transport.write_limit,
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
