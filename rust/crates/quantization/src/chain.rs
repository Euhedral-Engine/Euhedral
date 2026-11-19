//! A chain of stages on one lane's stream: each stage submits work that ends in a readback, and the retirement of that
//! work, delivered by the lane's [`Completions`], runs the next stage as a frame. Nothing waits on the device.
//!
//! The stream, the state the stages share and the lane's own parts travel with the chain from frame to frame. When
//! the chain ends, by a result or by a failure, [`Finished`] hands the lane back with the state, which the caller drops
//! off the lattice (freeing device and pinned memory waits for the device).

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
}

/// An error the chain can raise itself: the device refusing a submission, or a stage that has nowhere to go.
pub(crate) trait ChainError: From<GpuError> + Send + 'static {
    /// A chain that broke its own rules.
    fn broken(message: &str) -> Self;
}

/// State a chain used, handed back to the caller to drop off the lattice: freeing it waits for the device.
pub struct Scratch(Box<dyn Any + Send>);

impl Scratch {
    pub(crate) fn new(state: impl Any + Send) -> Scratch {
        Scratch(Box::new(state))
    }

    /// Frees the memory, waiting for the device: call it off the lattice.
    pub fn release(self) {
        drop(self.0);
    }
}

impl Debug for Scratch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Scratch").finish_non_exhaustive()
    }
}

/// What a chain hands to its `done` call, from the frame that ended it.
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
        mut stream,
        state: (mut link, _),
        error,
    }) = armed
    {
        // Work may still be in flight against the state: let it end before the state goes back.
        stream.quiesce();
        let cause = link.failure.take().unwrap_or_else(|| E::from(error));
        finish(stream, link, Err(cause));
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
