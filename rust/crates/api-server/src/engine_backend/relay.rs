//! The engine's listener, turned into the API's: tokens become text, prefill quanta and the end are passed on, and the
//! end is told exactly once whatever the engine does with the request.

use std::mem::take;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
pub(super) struct Hand {
    /// Set by the caller once the engine took the request: from then on a dropped relay ends the listener itself.
    pub(super) accepted: AtomicBool,
    pub(super) returned: Handoff<Box<dyn TextListener>>,
}

impl Hand {
    pub(super) fn new() -> Arc<Hand> {
        Arc::new(Hand {
            accepted: AtomicBool::new(false),
            returned: Handoff::empty(),
        })
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
    ) -> Relay {
        Relay {
            listener: Some(listener),
            stream: Some(stream),
            shape,
            cancel,
            hand,
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
        let tail = match self.stream.take() {
            Some(stream) if self.broken.is_none() => stream.finish(),
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
                cached_prompt_tokens: 0,
                token_ids: take(&mut self.tokens),
            }),
        };
        listener.finished(result);
    }
}

impl Drop for Relay {
    /// A relay dropped with its listener still waiting was refused or lost by the engine: the listener goes back to
    /// the caller, or, when the caller already took the engine's answer for acceptance, ends here.
    fn drop(&mut self) {
        let Some(listener) = self.listener.take() else {
            return;
        };
        self.hand.returned.send(listener);
        if self.hand.accepted.load(Ordering::SeqCst)
            && let Some(mut listener) = self.hand.returned.take()
        {
            listener.finished(Err(BackendError::Failed(
                "the engine dropped the generation".to_owned(),
            )));
        }
    }
}
