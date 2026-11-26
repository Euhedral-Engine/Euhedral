//! What the heads keep on the card between their calls, and the three pieces of bookkeeping around a call that must hold
//! whatever the card does: the buffers a failed call leaves are handed to the caller to free in stream order (never
//! dropped, which waits for the device), the MTP layer's shorter pass length is restored whether the layer ran or not,
//! and the resume file is unlisted in the manifest on disk before it is removed.
//!
//! The state is generic over the buffer, so that the contracts are tested with plain values; the card's is
//! `HeadState<DeviceBuffer>`.

use std::mem::take;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_inference_engine::Handoff;

use super::CalibrateError;
use super::driver::Then;
use super::manifest::Ledger;

/// What the MTP block holds on the card while its input projection is rounded and applied. A buffer is set the moment
/// it exists, so that a call that fails after it leaves it to be freed.
#[derive(Debug)]
pub struct MtpBuffers<B> {
    /// The embedding table.
    pub table: Option<B>,
    pub embedding_norm: Option<B>,
    pub hidden_norm: Option<B>,
    /// The input projection's weight, then its decoded weight.
    pub weight: Option<B>,
    pub decoded: Option<B>,
    /// The next tokens.
    pub ids: Option<B>,
    /// A batch of rows being normed, and their result.
    pub scratch: Option<B>,
    pub scratch_normed: Option<B>,
    /// A batch of stem rows.
    pub stem: Option<B>,
    /// The stem's Hessian.
    pub hessian: Option<B>,
}

impl<B> Default for MtpBuffers<B> {
    fn default() -> Self {
        MtpBuffers {
            table: None,
            embedding_norm: None,
            hidden_norm: None,
            weight: None,
            decoded: None,
            ids: None,
            scratch: None,
            scratch_normed: None,
            stem: None,
            hessian: None,
        }
    }
}

/// What the heads keep on the card between their calls.
#[derive(Debug)]
pub struct HeadState<B> {
    /// The final norm's weight.
    pub norm: Option<B>,
    /// A batch of states after the final norm, or after the MTP norm.
    pub normed: Option<B>,
    /// The output head's Hessian.
    pub hessian: Option<B>,
    /// The MTP norm's weight, until the draft head's Hessian is made.
    pub mtp_norm: Option<B>,
    /// The draft head's Hessian, when the MTP block made one.
    pub draft: Option<B>,
    /// The output head's weight, which the draft head's rows are taken from.
    pub weight: Option<B>,
    /// The MTP block's buffers, until its projection is applied.
    pub mtp: MtpBuffers<B>,
    /// The draft head's ids and gathered rows, until the heads are freed.
    pub gathered: Vec<B>,
}

impl<B> Default for HeadState<B> {
    fn default() -> Self {
        HeadState {
            norm: None,
            normed: None,
            hessian: None,
            mtp_norm: None,
            draft: None,
            weight: None,
            mtp: MtpBuffers::default(),
            gathered: Vec::new(),
        }
    }
}

impl<B> MtpBuffers<B> {
    /// Every buffer that is set.
    pub fn every(self) -> Vec<B> {
        [
            self.table,
            self.embedding_norm,
            self.hidden_norm,
            self.weight,
            self.decoded,
            self.ids,
            self.scratch,
            self.scratch_normed,
            self.stem,
            self.hessian,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl<B> HeadState<B> {
    /// Takes every buffer the state holds, leaving it empty.
    pub fn every(&mut self) -> Vec<B> {
        let mut buffers: Vec<B> = [
            self.norm.take(),
            self.normed.take(),
            self.hessian.take(),
            self.mtp_norm.take(),
            self.draft.take(),
            self.weight.take(),
        ]
        .into_iter()
        .flatten()
        .collect();
        buffers.extend(take(&mut self.mtp).every());
        buffers.append(&mut self.gathered);
        buffers
    }
}

/// After a failed call: hands every buffer the heads hold to `free`, which frees them in stream order. The buffers are
/// never dropped here, since a plain drop waits for the device. A state that is in use (a call is still running on it)
/// is left to that call; one that holds nothing calls nothing.
pub fn release_failed<B>(held: &Handoff<HeadState<B>>, free: impl FnOnce(Vec<B>)) {
    let Some(mut state) = held.take() else {
        return;
    };
    let buffers = state.every();
    held.send(state);
    if !buffers.is_empty() {
        free(buffers);
    }
}

/// Runs `run` with the pass length of `lengths` set to `during`, and sets it back to what it was when `run` ends, whether
/// it succeeded or not: the layers after the MTP layer, and a run that goes on after its failure, read the length of
/// the sequences. `run` gets the continuation to call when its work is over.
pub fn at_pass_length(lengths: &Arc<AtomicUsize>, during: usize, run: impl FnOnce(Then<()>), then: Then<()>) {
    let before = lengths.swap(during, Ordering::AcqRel);
    let restoring = Arc::clone(lengths);
    run(Box::new(move |result| {
        restoring.store(before, Ordering::Release);
        then(result);
    }));
}

/// Unlists the resume file in the manifest on disk (`save` writes the manifest's bytes), and only once that is saved
/// calls `remove` to remove the file: a manifest never lists a file that is gone. A failed save leaves the file and
/// the listing, and `remove` is not called; a failed removal leaves the file unlisted.
pub fn drop_resume_file(
    ledger: &Arc<Ledger>,
    save: impl FnOnce(Vec<u8>, Then<()>),
    remove: impl FnOnce(Then<()>) + Send + 'static,
    then: Then<()>,
) {
    ledger.resume_file_dropped(
        save,
        Box::new(move |unlisted: Result<(), CalibrateError>| match unlisted {
            Ok(()) => remove(then),
            Err(error) => then(Err(error)),
        }),
    );
}
