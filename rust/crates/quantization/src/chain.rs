//! A chain of stages on one lane's stream: each stage submits work that ends in a readback, and the retirement of that
//! work, delivered by the lane's [`Completions`], runs the next stage as a frame. Nothing waits on the device.
//!
//! The stream, the state the stages share and the lane's own parts travel with the chain from frame to frame. When
//! the chain ends, by a result or by a failure, [`Finished`] hands the lane back with the state. Dropping the state
//! gives its device buffers to the lane's queue (freed in stream order by the lane's next step) and its pinned staging
//! to the pool: nothing waits for the device.
//!
//! A chain that cannot arm its retirement ends at once, but the copies and launches it enqueued are still on the
//! stream and write into its pinned staging. That staging must not go back to the shared pool where another lane could
//! take it: the state is parked in the lane's [`Unsettled`], and the lane's next confirmed retirement (which the
//! stream's order puts behind that work) releases it. At shutdown the lane waits for the stream first.

use std::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_gpu::cuda::CudaStream;
use euhedral_gpu::{GpuError, OnceFrame, RetireError, RetireTicket, Stream, retire_into_with};
use euhedral_graph::Completions;

/// What a lane's parts must tell the chain.
pub(crate) trait Parts: Send + 'static {
    /// The completions that deliver the lane's retirements.
    fn completions(&self) -> &Arc<Completions>;
    /// The states of chains that could not arm their retirement.
    fn unsettled(&mut self) -> &mut Unsettled;
}

/// An error the chain can raise itself: the device refusing a submission, or a stage that has nowhere to go.
pub(crate) trait ChainError: From<GpuError> + Send + 'static {
    /// A chain that broke its own rules.
    fn broken(message: &str) -> Self;
}

/// State parked until the lane's stream has run the work queued before it.
#[derive(Default)]
pub(crate) struct Unsettled {
    held: Vec<Box<dyn Any + Send>>,
}

impl Unsettled {
    fn park(&mut self, state: Box<dyn Any + Send>) {
        self.held.push(state);
    }

    /// Releases what is parked: the stream has retired everything enqueued before now.
    fn release(&mut self) {
        self.held.clear();
    }

    /// How many failed chains' states are parked.
    pub(crate) fn len(&self) -> usize {
        self.held.len()
    }

    /// For a shutdown: calls `before_wait` if anything is parked, waits for the stream, then releases it. A frame never
    /// calls this.
    pub(crate) fn settle(&mut self, stream: &mut CudaStream, before_wait: impl FnOnce()) {
        if self.held.is_empty() {
            return;
        }
        before_wait();
        Stream::quiesce(stream);
        self.release();
    }
}

/// Memory a chain used. Dropping it gives its device buffers to the lane's queue (freed in stream order by the next
/// step on the lane, or by the lane's `reclaim`) and its pinned staging to the pool: nothing waits for the device. A
/// chain that failed to arm its retirement hands back an empty scratch: its state stays with the lane until a later
/// retirement confirms the stream has run the work still queued for it.
pub struct Scratch {
    _state: Box<dyn Any + Send>,
}

impl Scratch {
    pub(crate) fn new(state: impl Any + Send) -> Scratch {
        Scratch {
            _state: Box::new(state),
        }
    }
}

impl Debug for Scratch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Scratch").finish_non_exhaustive()
    }
}

/// What a chain hands to its `done` call, from the frame that ended it.
///
/// After a failed chain whose retirement could not be armed, `result` is an `Err` while the chain's kernels may still
/// read the caller's device memory and write what the caller gave them: it stays in use until the lane's unsettled count
/// is back to 0 or the lane's next step has been confirmed.
#[derive(Debug)]
pub struct Finished<L, T, E> {
    /// The lane, free again.
    pub lane: L,
    /// The memory the chain used.
    pub scratch: Scratch,
    /// The result, or the failure that ended the chain.
    pub result: Result<T, E>,
}

/// What a stage ends with.
pub(crate) enum Next<P, S, T, E> {
    /// It submitted work ending in a readback: run the next stage when that retires.
    Wait(Stage<P, S, T, E>),
    /// The result is complete.
    Done(T),
}

/// One stage: it may submit on the stream and read what the previous stage's readback brought.
pub(crate) type Stage<P, S, T, E> =
    Box<dyn FnOnce(&mut CudaStream, &mut P, &mut S) -> Result<Next<P, S, T, E>, E> + Send>;

type Done<L, T, E> = Box<dyn FnOnce(Finished<L, T, E>) + Send>;
type Rebuild<L, P> = fn(CudaStream, P) -> L;

struct Link<L, P, S, T, E> {
    parts: P,
    state: S,
    done: Done<L, T, E>,
    rebuild: Rebuild<L, P>,
    /// A stage failed with work possibly in flight: the chain waits for the retirement, then ends with this.
    failure: Option<E>,
}

/// Runs the first stage here, then each next one in the frame the lane's completions emit. `rebuild` puts the lane
/// back together for `done`.
pub(crate) fn start<L, P, S, T, E>(
    stream: CudaStream,
    parts: P,
    state: S,
    rebuild: Rebuild<L, P>,
    done: Done<L, T, E>,
    first: Stage<P, S, T, E>,
) where
    L: Send + 'static,
    P: Parts,
    S: Send + 'static,
    T: Send + 'static,
    E: ChainError,
{
    advance(
        stream,
        Link {
            parts,
            state,
            done,
            rebuild,
            failure: None,
        },
        first,
    );
}

fn advance<L, P, S, T, E>(mut stream: CudaStream, mut link: Link<L, P, S, T, E>, stage: Stage<P, S, T, E>)
where
    L: Send + 'static,
    P: Parts,
    S: Send + 'static,
    T: Send + 'static,
    E: ChainError,
{
    match stage(&mut stream, &mut link.parts, &mut link.state) {
        Ok(Next::Wait(next)) => arm(stream, link, Some(next)),
        Ok(Next::Done(value)) => finish(stream, link, Ok(value)),
        Err(error) => {
            link.failure = Some(error);
            arm(stream, link, None);
        }
    }
}

fn arm<L, P, S, T, E>(stream: CudaStream, link: Link<L, P, S, T, E>, next: Option<Stage<P, S, T, E>>)
where
    L: Send + 'static,
    P: Parts,
    S: Send + 'static,
    T: Send + 'static,
    E: ChainError,
{
    let completions = Arc::clone(link.parts.completions());
    let id = u64::from(stream.lane().0);
    let armed = retire_into_with(
        stream,
        (link, next),
        &completions,
        move |stream, (link, next), ticket| {
            let frame: Box<dyn Frame> = Box::new(OnceFrame::new(id, move || resume(stream, link, next, ticket)));
            frame
        },
    );
    if let Err(RetireError {
        stream,
        state: (mut link, _),
        error,
    }) = armed
    {
        // Work may still be queued behind this failure: the state stays with the lane until a retirement is confirmed.
        let cause = link.failure.take().unwrap_or_else(|| E::from(error));
        let Link {
            mut parts,
            state,
            done,
            rebuild,
            ..
        } = link;
        parts.unsettled().park(Box::new(state));
        done(Finished {
            lane: rebuild(stream, parts),
            scratch: Scratch::new(()),
            result: Err(cause),
        });
    }
}

fn resume<L, P, S, T, E>(
    mut stream: CudaStream,
    mut link: Link<L, P, S, T, E>,
    next: Option<Stage<P, S, T, E>>,
    ticket: RetireTicket,
) where
    L: Send + 'static,
    P: Parts,
    S: Send + 'static,
    T: Send + 'static,
    E: ChainError,
{
    let confirmed = stream.confirm_retired(ticket);
    // The stream has run everything enqueued before this retirement, what an earlier failed arm left included.
    link.parts.unsettled().release();
    if let Some(failure) = link.failure.take() {
        return finish(stream, link, Err(failure));
    }
    if let Some(error) = confirmed {
        return finish(stream, link, Err(E::from(error)));
    }
    match next {
        Some(stage) => advance(stream, link, stage),
        None => finish(stream, link, Err(E::broken("a chain resumed with no stage"))),
    }
}

fn finish<L, P, S, T, E>(stream: CudaStream, link: Link<L, P, S, T, E>, result: Result<T, E>)
where
    L: Send + 'static,
    P: Parts,
    S: Send + 'static,
    T: Send + 'static,
    E: ChainError,
{
    let Link {
        parts,
        state,
        done,
        rebuild,
        ..
    } = link;
    done(Finished {
        lane: rebuild(stream, parts),
        scratch: Scratch::new(state),
        result,
    });
}

/// Runs `body` as one submission, keeping its own error type.
pub(crate) fn submit<R, E: ChainError>(
    stream: &mut CudaStream,
    body: impl for<'a> FnOnce(&mut <CudaStream as Stream>::Enqueue<'a>) -> Result<R, E>,
) -> Result<R, E> {
    let mut failure = None;
    let outcome = stream.submit(|queue| match body(queue) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            failure = Some(error);
            Ok(None)
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    outcome?.ok_or_else(|| E::broken("a submission produced nothing"))
}

/// Arms a retirement on the stream that nobody collects, so the next chain's own cannot be armed and fails after it
/// enqueued its work. For tests of that path.
#[cfg(feature = "test-hooks")]
pub(crate) fn block_arming(stream: &mut CudaStream) -> Result<(), GpuError> {
    stream.arm_retirement()
}

/// Collects the retirement [`block_arming`] armed, once the stream has run up to it. Returns false when it has not
/// arrived yet.
#[cfg(feature = "test-hooks")]
pub(crate) fn unblock_arming(stream: &mut CudaStream) -> bool {
    match stream.poll_retired() {
        Some(ticket) => stream.confirm_retired(ticket).is_none(),
        None => false,
    }
}
