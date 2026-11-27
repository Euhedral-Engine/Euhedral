//! The generating engine of Flash-Next: one sequence, reached through a mailbox (as the dense engine's owner is).
//!
//! Role: [`FlashEngine`] owns the mailbox of the [`FlashOwner`] over the device's [`DeviceStepper`] and starts
//! requests on it the way the dense engine starts them: inline when the claim is free, else posted to the owner; a
//! request the owner refuses has its listener told. It is the unit the server will hold; nothing dispatches to it yet.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Duration;

use euhedral_data_structures::queues::MpscQueue;

use super::FlashError;
use super::device::{DeviceStepper, OwnerAddress};
use super::generate::{FlashOwner, Message, Rejected};
use super::startup::FlashModel;
use crate::generation::{AdmissionError, CancelHandle, Generation, GenerationRequest, Mailbox};

/// Messages a retiring quantum may have queued for the owner at once: one chunk is in flight, so a few suffice.
const INBOX: usize = 16;

fn closed_error() -> AdmissionError {
    AdmissionError::Engine("the engine is closed".to_string())
}

/// The engine's one sequence and the way to start requests on it.
pub struct FlashEngine {
    mailbox: Arc<Mailbox<FlashOwner<DeviceStepper>>>,
    /// Kept so the retiring quanta's address stays valid as long as the engine.
    _address: Arc<OwnerAddress>,
}

impl Debug for FlashEngine {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlashEngine").finish_non_exhaustive()
    }
}

impl FlashEngine {
    /// Allocates the sequence's device memory and the owner over it. Refuses with the allocation that failed.
    pub fn new(model: &FlashModel) -> Result<FlashEngine, FlashError> {
        let queue = MpscQueue::<Message>::new(INBOX);
        let address = OwnerAddress::new();
        let (stepper, limits) = DeviceStepper::new(model, &address)?;
        let owner = FlashOwner::new(stepper, queue.consumer(), limits);
        let mailbox = Arc::new(Mailbox::new(owner, queue.producer(), Generation::spawner(model.core())));
        address.set(&mailbox);
        Ok(FlashEngine {
            mailbox,
            _address: address,
        })
    }

    /// Starts `request`. `Ok` means the listener hears the outcome; an `Err` is the refusal, and the listener hears
    /// nothing. With the claim held (a listener calls this from its own callback) the request is posted and this
    /// returns `Ok`; a refusal then reaches the listener as a failed outcome.
    pub fn generate(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        let mut begin = Some((request, cancel));
        let inline = self.mailbox.try_with(|owner| match begin.take() {
            Some((request, cancel)) => owner.begin(request, cancel),
            None => Ok(()),
        });
        match inline {
            // The first quantum is started: its retirement is posted to the owner and drives the mailbox.
            Ok(Some(started)) => started.map_err(|rejected| rejected.error),
            Ok(None) => Err(closed_error()),
            Err(_) => {
                let Some((request, cancel)) = begin else {
                    return Err(closed_error());
                };
                self.mailbox.defer(move |owner| match owner {
                    Some(owner) => {
                        if let Err(rejected) = owner.begin(request, cancel) {
                            owner.reject(rejected);
                        }
                        Vec::new()
                    }
                    None => vec![Rejected::new(closed_error(), request.listener).into_finished()],
                });
                Ok(())
            }
        }
    }

    /// Cancels the running request and waits at most `bound` until it ended and its listener was told. `false` when the bound passed.
    pub fn stop(&self, bound: Duration) -> bool {
        self.mailbox.stopping().is_some_and(|idle| idle.wait(bound).is_ok())
    }

    /// Ends the engine's sequence and frees its device memory. Nothing may be running: with a request in flight the
    /// memory stays allocated and this says so. Drop the engine's stages (the model's) after this, before shutting
    /// the model down.
    pub fn close(self) {
        drop(self.mailbox.close());
    }
}

#[cfg(test)]
mod tests;
