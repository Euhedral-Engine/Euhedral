//! The device work of one window: its inputs decoded, every depth's forward pass, the draft head's cross-entropy and,
//! when training, the backward passes. Planned on the host from the addresses of the lane's buffers; nothing here
//! touches a device.

use super::error::RunError;
use super::settings::{HEAD_ROWS, Window};
use crate::mtp::{
    Drafter, Window as Span, Work, plan_cross_entropy, plan_float_to_bf16, plan_head_gradient, plan_head_logits,
    plan_q3_embedding_rows,
};
use crate::training::{plan_bf16_to_float, plan_zero};

/// What a window does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Forward, losses and gradients.
    Train,
    /// Forward and the draft head's argmaxes.
    Evaluate,
}

/// The device addresses and sizes a window's work is planned over.
#[derive(Clone, Debug)]
pub struct Addresses {
    /// The width of the hidden state.
    pub hidden: u32,
    /// The draft head's rows.
    pub draft_rows: u32,
    /// The drafts unrolled.
    pub depths: u32,
    /// What the head's gradient is scaled by: one over the lineages of an update.
    pub scale: f32,
    /// The draft head, BF16 `[draft_rows, hidden]`.
    pub head: u64,
    /// The Q3 embedding table, and the bytes in which its scales start.
    pub table: u64,
    pub scale_offset: u64,
    /// The base model's hidden rows of the window, BF16 `[rows, hidden]`, and as FP32.
    pub hidden_bf16: u64,
    pub hidden_input: u64,
    /// The input token of each draft, `[depths, rows]`, and the draft-head row each draft is trained towards.
    pub ids: u64,
    pub labels: u64,
    /// What the head returns: each draft's argmax and label log-probability `[depths, rows]`, and the summed loss of
    /// each depth (a double).
    pub predicted: u64,
    pub label_log_probabilities: u64,
    pub loss: u64,
    /// The embedding of each depth's input tokens, FP32 `[rows, hidden]`.
    pub embedded: Vec<u64>,
    /// A depth's output as BF16, the head's logits as FP32 for a chunk of rows, and as BF16.
    pub head_input: u64,
    pub logits: u64,
    pub logits_bf16: u64,
}

/// Plans `window` in `mode`: an unlabelled window runs depth 1 only (it writes the sequence's keys and values), a
/// labelled one every depth and the head's losses, and a training one the backward passes and the head's gradient.
pub fn plan(
    drafter: &mut Drafter,
    at: &Addresses,
    window: &Window,
    mode: Mode,
) -> Result<Vec<Work>, RunError> {
    let (count, hidden) = (window.count, at.hidden);
    let span = Span {
        position: u64::from(window.a),
        first_key: u64::from(window.first_key),
        count,
    };
    let rows = u64::from(count);
    let mut work = vec![Work::Steps(plan_bf16_to_float(
        at.hidden_bf16,
        rows * u64::from(hidden),
        at.hidden_input,
    )?)];
    for (depth, embedded) in at.embedded.iter().enumerate() {
        let ids = at.ids + 4 * depth as u64 * rows;
        work.push(Work::Steps(plan_q3_embedding_rows(
            at.table,
            ids,
            count,
            hidden,
            at.scale_offset,
            *embedded,
        )?));
    }
    let depths = if window.labelled { at.depths } else { 1 };
    let mut outputs = Vec::new();
    let mut input = at.hidden_input;
    for depth in 1..=depths {
        let forward = drafter.forward(depth, &span, input, at.embedded[depth as usize - 1])?;
        work.extend(forward.work);
        outputs.push(forward.output);
        input = forward.output;
    }
    if !window.labelled {
        return Ok(work);
    }
    let (training, scale) = (mode == Mode::Train, if mode == Mode::Train { at.scale } else { 0.0 });
    work.push(Work::Steps(plan_zero(at.loss, u64::from(depths) * 2)?));
    if training {
        work.extend(drafter.start_backward(&span)?);
    }
    for depth in 1..=depths {
        head(drafter, at, &mut work, depth, outputs[depth as usize - 1], count, scale)?;
    }
    if training {
        for depth in (1..=depths).rev() {
            work.extend(drafter.backward(depth, &span)?);
        }
    }
    Ok(work)
}

/// Depth `depth`'s cross-entropy against the draft head: argmax and label log-probabilities, the loss and, when `scale`
/// is not 0, the gradient into the depth's output gradient.
fn head(
    drafter: &Drafter,
    at: &Addresses,
    work: &mut Vec<Work>,
    depth: u32,
    output: u64,
    count: u32,
    scale: f32,
) -> Result<(), RunError> {
    let (hidden, draft_rows) = (u64::from(at.hidden), at.draft_rows);
    let offset = 4 * u64::from(depth - 1) * u64::from(count);
    work.push(Work::Steps(plan_float_to_bf16(
        output,
        u64::from(count) * hidden,
        at.head_input,
    )?));
    let mut first = 0;
    while first < count {
        let rows = HEAD_ROWS.min(count - first);
        let label = offset + 4 * u64::from(first);
        work.push(Work::Head(plan_head_logits(
            at.head,
            at.head_input + 2 * u64::from(first) * hidden,
            at.logits,
            rows,
            at.hidden,
            draft_rows,
        )?));
        work.push(Work::Steps(plan_cross_entropy(
            at.logits,
            rows,
            draft_rows,
            at.labels + label,
            scale,
            at.loss + 8 * u64::from(depth - 1),
            at.predicted + label,
            at.label_log_probabilities + label,
        )?));
        if scale != 0.0 {
            work.push(Work::Steps(plan_float_to_bf16(
                at.logits,
                u64::from(rows) * u64::from(draft_rows),
                at.logits_bf16,
            )?));
            let gradient = drafter
                .output_gradient(depth)
                .ok_or_else(|| RunError::Broken(format!("depth {depth} has no output gradient")))?;
            work.push(Work::Head(plan_head_gradient(
                at.head,
                at.logits_bf16,
                gradient + 4 * u64::from(first) * hidden,
                rows,
                at.hidden,
                draft_rows,
            )?));
        }
        first += rows;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;
    use crate::mtp::{Config, Shape};
    use crate::training::Step;

    const BASE: u64 = 0x10_0000;
    const HIDDEN: u32 = 64;
    const DRAFT_ROWS: u32 = 300;

    fn drafter(depths: u32) -> Drafter {
        let config = Config {
            shape: Shape {
                hidden: HIDDEN,
                intermediate: 96,
                heads: 4,
                kv_heads: 2,
                head_dim: 32,
                rotary_dim: 16,
                theta: 1e7,
                eps: 1e-6,
            },
            depths,
            rows: 300,
            max_keys: 300,
            max_positions: 600,
        };
        let mut drafter = Drafter::new(&config, BASE).unwrap();
        drafter.initialise().unwrap();
        drafter.measure().unwrap();
        drafter
    }

    fn addresses(depths: u32) -> Addresses {
        let at = |index: u64| 0x4000_0000 + index * 0x100_0000;
        Addresses {
            hidden: HIDDEN,
            draft_rows: DRAFT_ROWS,
            depths,
            scale: 0.25,
            head: at(0),
            table: at(1),
            scale_offset: 4096,
            hidden_bf16: at(2),
            hidden_input: at(3),
            ids: at(4),
            labels: at(5),
            predicted: at(6),
            label_log_probabilities: at(7),
            loss: at(8),
            embedded: (0..u64::from(depths)).map(|depth| at(10 + depth)).collect(),
            head_input: at(20),
            logits: at(21),
            logits_bf16: at(22),
        }
    }

    fn window(count: u32, labelled: bool) -> Window {
        Window {
            a: 0,
            count,
            first_key: 0,
            labelled,
        }
    }

    fn kernels(work: &[Work]) -> Vec<&'static str> {
        work.iter()
            .flat_map(|piece| match piece {
                Work::Steps(steps) => steps.iter().map(Step::kernel).collect(),
                Work::Product(_) | Work::Head(_) => vec!["gemm"],
            })
            .collect()
    }

    fn count(work: &[Work], name: &str) -> usize {
        kernels(work).iter().filter(|kernel| **kernel == name).count()
    }

    fn heads(work: &[Work]) -> usize {
        work.iter().filter(|piece| matches!(piece, Work::Head(_))).count()
    }

    #[test]
    fn an_unlabelled_window_runs_only_the_first_depths_forward() {
        let mut drafter = drafter(3);
        let work = plan(&mut drafter, &addresses(3), &window(5, false), Mode::Train).unwrap();
        assert_eq!(heads(&work), 0);
        assert_eq!(count(&work, "draft_cross_entropy"), 0);
        // The embeddings of every depth are staged, but only depth 1 is run.
        assert_eq!(count(&work, "q3_embedding_rows"), 3);
        let one = plan(&mut drafter, &addresses(3), &window(5, true), Mode::Evaluate).unwrap();
        assert!(one.len() > work.len());
    }

    #[test]
    fn an_evaluation_window_takes_losses_and_argmaxes_but_sends_no_gradient() {
        let mut drafter = drafter(2);
        let work = plan(&mut drafter, &addresses(2), &window(5, true), Mode::Evaluate).unwrap();
        assert_eq!(count(&work, "draft_cross_entropy"), 2);
        assert_eq!(heads(&work), 2, "one logits product per depth and no gradient product");
    }

    #[test]
    fn a_training_window_sends_the_head_gradient_back_and_runs_the_backward_passes() {
        let mut drafter = drafter(2);
        let evaluation = plan(&mut drafter, &addresses(2), &window(5, true), Mode::Evaluate).unwrap();
        let training = plan(&mut drafter, &addresses(2), &window(5, true), Mode::Train).unwrap();
        assert_eq!(heads(&training), 4, "a logits and a gradient product per depth");
        assert_eq!(count(&training, "draft_cross_entropy"), 2);
        assert!(training.len() > evaluation.len());
    }

    #[test]
    fn the_head_works_in_chunks_of_at_most_the_head_rows() {
        let mut drafter = drafter(1);
        let count_rows = 2 * HEAD_ROWS + 1;
        let work = plan(&mut drafter, &addresses(1), &window(count_rows, true), Mode::Train).unwrap();
        assert_eq!(count(&work, "draft_cross_entropy"), 3);
        assert_eq!(heads(&work), 6);
        let mut one = drafter;
        let single = plan(&mut one, &addresses(1), &window(HEAD_ROWS, true), Mode::Train).unwrap();
        assert_eq!(count(&single, "draft_cross_entropy"), 1);
    }

    #[test]
    fn a_window_the_drafter_cannot_hold_is_refused() {
        let mut drafter = drafter(1);
        assert!(plan(&mut drafter, &addresses(1), &window(301, true), Mode::Train).is_err());
        assert!(plan(&mut drafter, &addresses(1), &window(0, true), Mode::Train).is_err());
    }
}
