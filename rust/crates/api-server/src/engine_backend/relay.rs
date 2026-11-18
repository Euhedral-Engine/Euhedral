//! The engine's listener, turned into the API's: tokens become text, prefill quanta and the end are passed on, and the
//! end is told exactly once whatever the engine does with the request.

use std::mem::take;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use euhedral_inference_engine::Handoff;
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationListener, GenerationOutcome, QuantumEvent,
};
use euhedral_inference_engine::stages::QuantumKind;

use super::{Codec, TextStream};
use crate::backend::{BackendError, GenerationResult, TextListener};

/// The listener's way back out of an engine that refused the request: the engine drops the relay without telling it,
/// so the relay gives the API's listener to whoever offered the request, who ends it.
///
/// One state decides who ends the listener: `looked`, set by the caller once it has the engine's answer. A dropped
/// relay puts its listener in `returned`, then reads `looked`; the caller sets `looked`, then takes from `returned`.
/// Both orders are sequentially consistent, so one of the two sees the other and the take makes the end happen once.
pub(super) struct Hand {
    /// Set by the caller once the engine answered the offer.
    looked: AtomicBool,
    /// Why the engine refused, set before `looked`; absent when it took the request.
    refusal: OnceLock<BackendError>,
    returned: Handoff<Box<dyn TextListener>>,
}

impl Hand {
    pub(super) fn new() -> Arc<Hand> {
        Arc::new(Hand {
            looked: AtomicBool::new(false),
            refusal: OnceLock::new(),
            returned: Handoff::empty(),
        })
    }

    /// The caller has the engine's answer: ends the listener now if the engine already let go of it, else leaves that
    /// to the relay's drop.
    pub(super) fn answered(&self, refusal: Option<BackendError>) {
        if let Some(refusal) = refusal {
            // Only the caller sets it, once.
            let _ = self.refusal.set(refusal);
        }
        self.looked.store(true, Ordering::SeqCst);
        self.end_returned();
    }

    fn end_returned(&self) {
        if let Some(mut listener) = self.returned.take() {
            let error = self
                .refusal
                .get()
                .cloned()
                .unwrap_or_else(|| BackendError::Failed("the engine dropped the generation".to_owned()));
            listener.finished(Err(error));
        }
    }
}

/// What a generation needs of the model to turn its tokens into a result.
pub(super) struct Shape {
    pub(super) stop_tokens: Vec<u32>,
    pub(super) think_end: Option<u32>,
    pub(super) reasoning: bool,
}

impl Shape {
    pub(super) fn of(codec: &dyn Codec, reasoning: bool) -> Shape {
        Shape {
            stop_tokens: codec.stop_tokens(),
            think_end: codec.think_end(),
            reasoning,
        }
    }
}

pub(super) struct Relay {
    listener: Option<Box<dyn TextListener>>,
    stream: Option<Box<dyn TextStream>>,
    shape: Shape,
    cancel: CancelHandle,
    hand: Arc<Hand>,
    /// The position the first prefill quantum started at, which is what the prefix cache restored: set once, before
    /// the first token, for the generation to report while it runs.
    restored: Arc<AtomicUsize>,
    prefilling: bool,
    tokens: Vec<u32>,
    /// Why decoding a token failed; the generation is cancelled and ends as a failure.
    broken: Option<String>,
}

impl Relay {
    pub(super) fn new(
        listener: Box<dyn TextListener>,
        stream: Box<dyn TextStream>,
        shape: Shape,
        cancel: CancelHandle,
        hand: Arc<Hand>,
        restored: Arc<AtomicUsize>,
    ) -> Relay {
        Relay {
            listener: Some(listener),
            stream: Some(stream),
            shape,
            cancel,
            hand,
            restored,
            prefilling: false,
            tokens: Vec::new(),
            broken: None,
        }
    }

    /// The tokens before `</think>`; when it never came, every token but a terminator.
    fn reasoning_tokens(&self) -> usize {
        if !self.shape.reasoning {
            return 0;
        }
        let ended = self
            .shape
            .think_end
            .and_then(|end| self.tokens.iter().position(|id| *id == end));
        ended.unwrap_or_else(|| self.tokens.len() - usize::from(self.stopped()))
    }

    fn stopped(&self) -> bool {
        self.tokens.last().is_some_and(|id| self.shape.stop_tokens.contains(id))
    }
}

impl GenerationListener for Relay {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        if self.broken.is_some() {
            return;
        }
        self.tokens.push(id);
        if self.shape.stop_tokens.contains(&id) {
            return;
        }
        let Some(stream) = self.stream.as_mut() else {
            return;
        };
        match stream.push(id) {
            Ok(text) if text.is_empty() => {}
            Ok(text) => {
                if let Some(listener) = self.listener.as_mut() {
                    listener.text(&text);
                }
            }
            Err(reason) => {
                self.broken = Some(reason);
                self.cancel.cancel();
            }
        }
    }

    fn quantum(&mut self, event: QuantumEvent) {
        if event.kind == QuantumKind::Prefill && !self.prefilling {
            self.prefilling = true;
            self.restored.store(
                usize::try_from(event.start_position).unwrap_or(usize::MAX),
                Ordering::Release,
            );
        }
        if event.kind == QuantumKind::Prefill
            && let Some(listener) = self.listener.as_mut()
        {
            listener.prefilled();
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let Some(mut listener) = self.listener.take() else {
            return;
        };
        // Held-back bytes are text only when the generation ran to its end.
        let ended = matches!(
            outcome.reason,
            FinishReason::Stop | FinishReason::Length | FinishReason::ContextFull
        );
        let tail = match self.stream.take() {
            Some(stream) if ended && self.broken.is_none() => stream.finish(),
            _ => String::new(),
        };
        if !tail.is_empty() {
            listener.text(&tail);
        }
        let result = match (&self.broken, outcome.reason) {
            (Some(reason), _) => Err(BackendError::Failed(reason.clone())),
            (None, FinishReason::Failed(reason)) => Err(BackendError::Failed(reason)),
            (None, _) => Ok(GenerationResult {
                completion_tokens: self.tokens.len(),
                stop_token_reached: self.stopped(),
                reasoning_tokens: self.reasoning_tokens(),
                cached_prompt_tokens: outcome.cached_prompt_tokens,
                token_ids: take(&mut self.tokens),
            }),
        };
        listener.finished(result);
    }
}

impl Drop for Relay {
    /// A relay dropped with its listener still waiting was refused or lost by the engine: the listener goes back to
    /// the caller, or, when the caller already has the engine's answer, ends here with the right error.
    fn drop(&mut self) {
        let Some(listener) = self.listener.take() else {
            return;
        };
        self.hand.returned.send(listener);
        if self.hand.looked.load(Ordering::SeqCst) {
            self.hand.end_returned();
        }
    }
}
