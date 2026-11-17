//! Starting and stopping the server on a lattice.

use std::fmt::{self, Debug, Formatter};
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::generics::LatticeSource;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::Completions;
use euhedral_io::AsyncIo;
use euhedral_net::{Counters, ServerHandle, serve};

use crate::backend::Backend;
use crate::limits::Limits;
use crate::metrics::Metrics;
use crate::router::Router;

/// What a server needs to start.
#[derive(Clone)]
pub struct ApiConfig {
    /// The address to listen on; port 0 takes a free one (see [`ApiServer::local_addr`]).
    pub address: SocketAddr,
    pub limits: Limits,
    /// What the generating routes use. Its identity is the model name the lists and `/health` show, and its readiness
    /// is what `/health` and those routes report.
    pub backend: Arc<dyn Backend>,
    /// When the model was created, in seconds since the epoch, as the model lists show it.
    pub created_at: i64,
    /// The sockets' engine, when the caller has one to share: it must be attached to the lattice already, and the
    /// caller completes it. `None`: the server opens one, attaches it and completes it on release.
    pub io: Option<Arc<AsyncIo>>,
}

impl Debug for ApiConfig {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ApiConfig")
            .field("address", &self.address)
            .field("limits", &self.limits)
            .field("created_at", &self.created_at)
            .finish_non_exhaustive()
    }
}

/// A running server. Stop it, wait to be told, then [`release`](ApiServer::release) it before the lattice closes.
pub struct ApiServer {
    handle: ServerHandle,
    local_addr: SocketAddr,
    sink: Arc<QueueIngestSink>,
    timers: Arc<Completions>,
    // The sockets' engine, held so that it outlives the server (the transport holds it weakly); `own_io` says the
    // server opened it and so completes it.
    io: Arc<AsyncIo>,
    own_io: bool,
    metrics: Arc<Metrics>,
}

impl Debug for ApiServer {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ApiServer")
            .field("local_addr", &self.local_addr)
            .finish_non_exhaustive()
    }
}

/// Starts the server on `lattice`: it binds the address, attaches the sockets' engine (unless the config brings one),
/// the sink and the timers to the lattice and attaches the listener as a source. Every step of the server is a frame
/// of the lattice. If it fails, what it attached is completed and nothing is left running.
pub fn start(lattice: &Arc<ControlPlaneLattice>, config: ApiConfig) -> io::Result<ApiServer> {
    let listener = TcpListener::bind(config.address)?;
    let local_addr = listener.local_addr()?;
    let (io, own_io) = match config.io {
        Some(io) => (io, false),
        None => (AsyncIo::open()?, true),
    };
    let sink = Arc::new(QueueIngestSink::new());
    let timers = Completions::without_lanes();
    let undo = |attached_io: bool| {
        sink.complete_gracefully();
        timers.complete();
        if attached_io && own_io {
            io.complete();
        }
    };
    let attach = |result: Result<_, _>| result.map(|_| ()).map_err(|e| io::Error::other(format!("{e:?}")));
    if own_io && let Err(error) = attach(lattice.add_upstream(io.clone())) {
        undo(false);
        return Err(error);
    }
    if let Err(error) =
        attach(lattice.add_upstream(sink.delegate())).and_then(|()| attach(lattice.add_upstream(timers.clone())))
    {
        undo(true);
        return Err(error);
    }
    let metrics = Arc::new(Metrics::new(
        config.backend.engine_report().speculative_depth,
        Metrics::wall_clock_ms(),
    ));
    let router = Router::new(config.backend, config.created_at, config.limits, metrics.clone());
    let handle = match serve(
        lattice.clone(),
        io.clone(),
        sink.clone(),
        listener,
        config.limits.transport(),
        Arc::new(router),
        timers.clone(),
    ) {
        Ok(handle) => handle,
        Err(error) => {
            undo(true);
            return Err(error);
        }
    };
    Ok(ApiServer {
        handle,
        local_addr,
        sink,
        timers,
        io,
        own_io,
        metrics,
    })
}

impl ApiServer {
    /// The address the server listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The registry the routes export: what the request code records into.
    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
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

    /// Completes everything the server opened: its sink, its timers and, if it opened it, the sockets' engine. Call it
    /// once the stop was reported (no operation is pending then) and before the lattice closes.
    pub fn release(self) {
        self.sink.complete_gracefully();
        self.timers.complete();
        if self.own_io {
            self.io.complete();
        }
    }
}
