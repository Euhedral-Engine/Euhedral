//! Starting and stopping the server on a lattice.

use std::fmt::{self, Debug, Formatter};
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::Completions;
use euhedral_io::AsyncIo;
use euhedral_net::{Counters, ServerHandle, serve};

use crate::backend::Backend;
use crate::limits::Limits;
use crate::router::Router;

/// What a server needs to start.
#[derive(Clone)]
pub struct ApiConfig {
    /// The address to listen on; port 0 takes a free one (see [`ApiServer::local_addr`]).
    pub address: SocketAddr,
    pub limits: Limits,
    /// What the generating routes use, and whose readiness `/health` and those routes report.
    pub backend: Arc<dyn Backend>,
    /// The public model name: what the model lists and `/health` show.
    pub model_name: String,
    /// When the model was created, in seconds since the epoch, as the model lists show it.
    pub created_at: i64,
}

impl Debug for ApiConfig {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ApiConfig")
            .field("address", &self.address)
            .field("limits", &self.limits)
            .field("model_name", &self.model_name)
            .field("created_at", &self.created_at)
            .finish_non_exhaustive()
    }
}

/// A running server. Stop it, wait to be told, then [`release`](ApiServer::release) it before the lattice closes.
pub struct ApiServer {
    handle: ServerHandle,
    local_addr: SocketAddr,
    sink: Arc<QueueIngestSink>,
    // Held so that the engine outlives the server: the transport holds it weakly.
    _io: Arc<AsyncIo>,
}

impl Debug for ApiServer {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ApiServer")
            .field("local_addr", &self.local_addr)
            .finish_non_exhaustive()
    }
}

/// Starts the server on `lattice`: it binds the address, attaches the sockets' engine, the sink and the timers to the
/// lattice and attaches the listener as a source. Every step of the server is a frame of the lattice.
pub fn start(lattice: &Arc<ControlPlaneLattice>, config: ApiConfig) -> io::Result<ApiServer> {
    let listener = TcpListener::bind(config.address)?;
    let local_addr = listener.local_addr()?;
    let io = AsyncIo::open()?;
    let sink = Arc::new(QueueIngestSink::new());
    let timers = Completions::without_lanes();
    let attach = |result: Result<_, _>| result.map(|_| ()).map_err(|e| io::Error::other(format!("{e:?}")));
    attach(lattice.add_upstream(io.clone()))?;
    attach(lattice.add_upstream(sink.delegate()))?;
    attach(lattice.add_upstream(timers.clone()))?;
    let router = Router::new(config.backend, config.model_name, config.created_at, config.limits);
    let handle = serve(
        lattice.clone(),
        io.clone(),
        sink.clone(),
        listener,
        config.limits.transport(),
        Arc::new(router),
        timers,
    )?;
    Ok(ApiServer {
        handle,
        local_addr,
        sink,
        _io: io,
    })
}

impl ApiServer {
    /// The address the server listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// What the transport counted so far.
    pub fn counters(&self) -> Counters {
        self.handle.counters()
    }

    /// Stops accepting, lets the requests in flight finish and offers `stopped` when the server is done. Returns at
    /// once.
    pub fn stop(&self, stopped: Box<dyn Frame>) {
        self.handle.stop(stopped);
    }

    /// Lets go of the server's sink once the stop was reported. Call it before the lattice closes.
    pub fn release(self) {
        self.sink.complete_gracefully();
    }
}
