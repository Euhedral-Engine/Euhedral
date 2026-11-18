//! What a request may ask of the tokens it selects: a constraint that says which tokens may be chosen at each step.
//!
//! The constraint moves with the request's frames, never shared: the frame that prepares the next step's mask owns it,
//! then the frame that selects from the logits row owns it and commits the token it chose, and the owner holds it in
//! between.

use std::fmt::{self, Formatter};

use thiserror::Error;

use crate::sampling::TokenFilter;

/// Why a constraint could not prepare a step or commit a token.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ConstraintError {
    message: String,
}

impl ConstraintError {
    pub fn new(message: impl Into<String>) -> ConstraintError {
        ConstraintError {
            message: message.into(),
        }
    }
}

/// A request-local restriction on which tokens the next selection may choose.
///
/// [`TokenFilter::allows`] (and [`TokenFilter::mask_disallowed`]) answer for the step after the last accepted token and
/// do not change state. A constraint that must compute something to answer does it in [`TokenConstraint::prepare`],
/// which runs while the device computes the logits the filter is then applied to. Both run on a worker, one step after
/// another, and never at once.
pub trait TokenConstraint: TokenFilter + Send {
    /// Computes what `allows` answers from, for the next selection: called once per step, before it.
    fn prepare(&mut self) -> Result<(), ConstraintError> {
        Ok(())
    }

    /// Commits the token the selection chose; an error when the constraint does not allow it.
    fn accept(&mut self, id: u32) -> Result<(), ConstraintError>;
}

/// Output that opens inside the model's think block: free reasoning until the end-of-think token, then an answer that
/// may follow a constraint of its own.
///
/// The reasoning may end at any point, and must once `budget` reasoning tokens were generated. With an answer
/// constraint a terminator cannot end the reasoning, since no answer would exist; the answer constraint decides
/// everything after the end-of-think token. Without one the answer is free.
pub struct ReasoningConstraint {
    end_think: u32,
    eos: Vec<u32>,
    budget: Option<u32>,
    answer: Option<Box<dyn TokenConstraint>>,
    answering: bool,
    reasoning_tokens: u32,
}

impl ReasoningConstraint {
    /// `end_think` is the id that closes the think block, `eos` the generation terminators, `budget` the most reasoning
    /// tokens (`None` leaves them uncapped), `answer` the constraint of what follows (`None` is free).
    pub fn new(
        end_think: u32,
        eos: Vec<u32>,
        budget: Option<u32>,
        answer: Option<Box<dyn TokenConstraint>>,
    ) -> ReasoningConstraint {
        ReasoningConstraint {
            end_think,
            eos,
            budget,
            answer,
            answering: false,
            reasoning_tokens: 0,
        }
    }

    /// Reasoning tokens committed so far, the end-of-think token excluded.
    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    fn spent(&self) -> bool {
        self.budget.is_some_and(|budget| self.reasoning_tokens >= budget)
    }
}

impl TokenFilter for ReasoningConstraint {
    fn allows(&self, id: u32) -> bool {
        if self.answering {
            return self.answer.as_ref().is_none_or(|answer| answer.allows(id));
        }
        if id == self.end_think {
            return true;
        }
        if self.spent() {
            return false;
        }
        self.answer.is_none() || !self.eos.contains(&id)
    }

    fn mask_disallowed(&self, row: &mut [f32]) {
        if self.answering {
            if let Some(answer) = &self.answer {
                answer.mask_disallowed(row);
            }
            return;
        }
        if self.spent() {
            for (id, logit) in (0..).zip(row.iter_mut()) {
                if id != self.end_think {
                    *logit = f32::NEG_INFINITY;
                }
            }
            return;
        }
        if self.answer.is_none() {
            return;
        }
        for &id in &self.eos {
            if let Some(logit) = row.get_mut(id as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

impl TokenConstraint for ReasoningConstraint {
    fn prepare(&mut self) -> Result<(), ConstraintError> {
        match &mut self.answer {
            Some(answer) if self.answering => answer.prepare(),
            _ => Ok(()),
        }
    }

    fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
        if !self.allows(id) {
            return Err(ConstraintError::new("token violates the reasoning constraint"));
        }
        if self.answering {
            if let Some(answer) = &mut self.answer {
                answer.accept(id)?;
            }
        } else if id == self.end_think {
            self.answering = true;
        } else {
            self.reasoning_tokens += 1;
        }
        Ok(())
    }
}

impl fmt::Debug for ReasoningConstraint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReasoningConstraint")
            .field("answering", &self.answering)
            .field("reasoning_tokens", &self.reasoning_tokens)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END: u32 = 9;
    const EOS: u32 = 8;
    const VOCABULARY: usize = 12;

    /// An answer that allows the ids of its set and counts what it was told.
    struct Only(Vec<u32>, Vec<u32>, u32);

    impl TokenFilter for Only {
        fn allows(&self, id: u32) -> bool {
            self.0.contains(&id)
        }
    }

    impl TokenConstraint for Only {
        fn prepare(&mut self) -> Result<(), ConstraintError> {
            self.2 += 1;
            Ok(())
        }

        fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
            if !self.allows(id) {
                return Err(ConstraintError::new("not allowed"));
            }
            self.1.push(id);
            Ok(())
        }
    }

    fn accept_all(constraint: &mut ReasoningConstraint, ids: &[u32]) {
        for &id in ids {
            assert!(constraint.allows(id), "token {id} rejected");
            constraint.accept(id).unwrap();
        }
    }

    #[test]
    fn reasoning_is_free_then_the_answer_follows_its_constraint() {
        let answer = Only(vec![1, 2, EOS], Vec::new(), 0);
        let mut constraint = ReasoningConstraint::new(THINK_END, vec![EOS], None, Some(Box::new(answer)));
        let reasoning = [3, 4, 5, 6, 7];
        accept_all(&mut constraint, &reasoning);
        assert!(!constraint.allows(EOS), "no terminator before an answer exists");
        let mut row = vec![0.0f32; VOCABULARY];
        constraint.mask_disallowed(&mut row);
        assert_eq!(row[EOS as usize], f32::NEG_INFINITY);
        assert_eq!(row[THINK_END as usize], 0.0);
        assert!(constraint.allows(THINK_END));
        constraint.accept(THINK_END).unwrap();
        assert!(!constraint.allows(3), "the answer follows its constraint");
        accept_all(&mut constraint, &[1, 2]);
        assert!(constraint.allows(EOS));
        assert_eq!(constraint.reasoning_tokens(), reasoning.len() as u32);
    }

    #[test]
    fn the_budget_forces_the_end_of_reasoning() {
        let mut constraint = ReasoningConstraint::new(THINK_END, vec![EOS], Some(3), None);
        accept_all(&mut constraint, &[1, 2, 3]);
        assert!(!constraint.allows(4));
        assert!(!constraint.allows(EOS));
        assert!(constraint.accept(4).is_err());
        let mut row = vec![0.0f32; VOCABULARY];
        constraint.mask_disallowed(&mut row);
        for (id, logit) in row.iter().enumerate() {
            let expected = if id as u32 == THINK_END { 0.0 } else { f32::NEG_INFINITY };
            assert_eq!(*logit, expected, "id {id}");
        }
        constraint.accept(THINK_END).unwrap();
        assert!(constraint.allows(4), "the answer is free without a constraint");
        assert!(constraint.allows(EOS));
    }

    #[test]
    fn without_an_answer_a_terminator_may_end_the_reasoning() {
        let constraint = ReasoningConstraint::new(THINK_END, vec![EOS], Some(10), None);
        assert!(constraint.allows(EOS));
        let mut row = vec![0.0f32; VOCABULARY];
        constraint.mask_disallowed(&mut row);
        assert!(row.iter().all(|logit| *logit == 0.0), "nothing is masked");
    }

    #[test]
    fn the_answer_prepares_only_once_it_decides() {
        let mut constraint =
            ReasoningConstraint::new(THINK_END, vec![EOS], None, Some(Box::new(Only(vec![1], Vec::new(), 0))));
        constraint.prepare().unwrap();
        constraint.accept(THINK_END).unwrap();
        constraint.prepare().unwrap();
        constraint.accept(1).unwrap();
        assert!(
            constraint.accept(2).is_err(),
            "the answer's refusal is the constraint's"
        );
    }
}
