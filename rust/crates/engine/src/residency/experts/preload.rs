//! The startup fill of the tier: every slot an assignment gave a record is read from the artifact into its slot.
//!
//! `readers` chains run side by side. Each takes the next record from a shared counter, reads it, and on its end takes
//! another: nothing waits, and no thread is made. The fill ends once, through `done`, with the bytes read, or the first
//! failure (the chains stop taking records after one).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use super::arena::HostArena;
use euhedral_data_structures::sync::ClaimCell;

use super::layout::{ExpertError, ExpertId};
use super::source::{FileRecordSource, ReadLoad, ReadOutcome, Started};

/// What a preload ended with.
pub type PreloadDone = Box<dyn FnOnce(Result<u64, ExpertError>) + Send>;

struct Fill {
    items: Vec<(ExpertId, usize)>,
    source: Arc<FileRecordSource>,
    arena: Arc<HostArena>,
    next: AtomicUsize,
    /// Chains still running.
    chains: AtomicUsize,
    bytes: AtomicU64,
    failure: OnceLock<ExpertError>,
    cancel: Arc<AtomicBool>,
    done: ClaimCell<Option<PreloadDone>>,
}

/// Starts reading `items` (a record and the tier slot it goes to) into `arena` with `readers` chains. Refuses no
/// reader; with no items `done` is called at once with 0.
pub fn preload(
    source: Arc<FileRecordSource>,
    arena: Arc<HostArena>,
    items: Vec<(ExpertId, usize)>,
    readers: usize,
    done: PreloadDone,
) -> Result<(), ExpertError> {
    let chains = readers.clamp(1, items.len().max(1));
    if items.is_empty() {
        done(Ok(0));
        return Ok(());
    }
    let fill = Arc::new(Fill {
        items,
        source,
        arena,
        next: AtomicUsize::new(0),
        chains: AtomicUsize::new(chains),
        bytes: AtomicU64::new(0),
        failure: OnceLock::new(),
        cancel: Arc::new(AtomicBool::new(false)),
        done: ClaimCell::new(Some(done)),
    });
    for _ in 0..chains {
        advance(&fill);
    }
    Ok(())
}

/// Takes the chain's next record and starts it, or ends the chain.
fn advance(fill: &Arc<Fill>) {
    let index = fill.next.fetch_add(1, Ordering::AcqRel);
    let Some(&(id, slot)) = fill.items.get(index).filter(|_| fill.failure.get().is_none()) else {
        return end_chain(fill);
    };
    let chained = Arc::clone(fill);
    let started = fill.arena.write(slot).and_then(|destination| {
        fill.source.begin(ReadLoad {
            id,
            destination,
            copy_to: None,
            cancel: Arc::clone(&fill.cancel),
            done: Box::new(move |outcome| {
                match outcome {
                    ReadOutcome::Complete => {
                        let bytes = chained.source.layout().record(id).map_or(0, |record| record.bytes);
                        chained.bytes.fetch_add(bytes, Ordering::AcqRel);
                    }
                    ReadOutcome::Failed(error) => {
                        let _ = chained.failure.set(error);
                    }
                    ReadOutcome::Cancelled | ReadOutcome::Lost => {
                        let _ = chained.failure.set(ExpertError::Unfinished {
                            bank: chained.source.layout().bank_name(id.bank).to_owned(),
                            expert: id.expert,
                        });
                    }
                }
                advance(&chained);
            }),
        })
    });
    match started {
        Ok(Started::Reading) => {}
        // Every part had ended: the frame finishes the load, which continues this chain in its `done`.
        Ok(Started::Frame(mut frame)) => {
            let _ = frame.execute();
        }
        Err(error) => {
            let _ = fill.failure.set(error);
            end_chain(fill);
        }
    }
}

fn end_chain(fill: &Arc<Fill>) {
    if fill.chains.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    let Some(mut claim) = fill.done.try_claim() else {
        return;
    };
    if let Some(done) = claim.take() {
        drop(claim);
        done(match fill.failure.get() {
            Some(error) => Err(error.clone()),
            None => Ok(fill.bytes.load(Ordering::Acquire)),
        });
    }
}
