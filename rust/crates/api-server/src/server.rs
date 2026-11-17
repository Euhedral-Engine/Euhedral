//! Starting and stopping the server on a lattice.

use std::fmt::{self, Debug, Formatter};
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::generics::LatticeSource;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::Completions;
use euhedral_io::AsyncIo;
use euhedral_net::{Counters, ServerHandle, serve};

use crate::admission::Admission;
use crate::backend::Backend;
use crate::frames::once_or_refused;
use crate::limits::Limits;
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
    admission: Admission,
    // The sockets' engine, held so that it outlives the server (the transport holds it weakly); `own_io` says the
    // server opened it and so completes it.
    io: Arc<AsyncIo>,
    own_io: bool,
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
    let admission = Admission::new(config.backend.clone(), &config.limits, sink.clone(), timers.clone());
    let router = Router::new(config.backend, config.created_at, config.limits);
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
        admission,
        io,
        own_io,
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

    /// The admission of generations the generating routes use.
    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    /// Stops the server and offers `stopped` when it is done; returns at once.
    ///
    /// Admission stops first: the requests waiting for a slot are refused, the running generations are interrupted and
    /// the stop waits for them up to the shutdown limit. At the same time the transport stops accepting and lets the
    /// requests in flight finish, which the refusals and the interruptions bring about. `stopped` is offered once both
    /// are done.
    pub fn stop(&self, stopped: Box<dyn Frame>) {
        let join = Arc::new(Join {
            remaining: AtomicUsize::new(2),
            stopped: ClaimCell::new(Some(stopped)),
            sink: self.sink.clone(),
        });
        let admitted = join.clone();
        self.admission.stop(move || admitted.arrive());
        self.handle.stop(once_or_refused(move |_| join.arrive()));
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

/// Offers `stopped` when the last of the two stops arrives.
struct Join {
    remaining: AtomicUsize,
    /// Taken by the last arrival only, so it is never contended.
    stopped: ClaimCell<Option<Box<dyn Frame>>>,
    sink: Arc<QueueIngestSink>,
}

impl Join {
    fn arrive(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if let Some(mut claim) = self.stopped.try_claim()
            && let Some(stopped) = claim.take()
        {
            drop(self.sink.offer(stopped));
        }
    }
}
