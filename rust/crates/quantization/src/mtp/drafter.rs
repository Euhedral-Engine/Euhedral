//! The drafter: its parameters, its arena and the planners of its forward and backward passes.

use super::layout::{self, Arena, Config, Parameter, Shape};
use super::plan::{
    Lineage, Softmax, plan_absolute_maximum, plan_copy_columns, plan_lineage_dot, plan_lineage_gather,
    plan_lineage_spread, plan_nvfp4_round, plan_softmax, plan_softmax_backward,
};
use super::{DrafterError, Work};
use crate::training::{
    AdamSettings, Gemm, Heads, Matrices, Step, TrainError, plan_adam, plan_copy, plan_gate_backward, plan_gate_forward,
    plan_linear, plan_linear_input_gradient, plan_product, plan_product_left_transposed, plan_product_transposed,
    plan_qk_norm_rope_backward, plan_qk_norm_rope_forward, plan_rms_norm_backward, plan_rms_norm_forward,
    plan_swiglu_backward, plan_swiglu_forward, plan_zero,
};

/// The offset of the norms' weight: the layer's norms use `1 + w`.
const NORM_OFFSET: f32 = 1.0;
/// The AdamW decay rates and epsilon of the parameters' step.
const FIRST_DECAY: f32 = 0.9;
const SECOND_DECAY: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

// Indices into the parameters, which are in parameter order.
const INPUT_PROJECTION: usize = 0;
const QUERY_KEY_GATE_VALUE: usize = 1;
const OUTPUT: usize = 2;
const GATE_UP: usize = 3;
const DOWN: usize = 4;
const EMBEDDING_NORM: usize = 5;
const HIDDEN_NORM: usize = 6;
const INPUT_NORM: usize = 7;
const POST_ATTENTION_NORM: usize = 8;
const QUERY_NORM: usize = 9;
const KEY_NORM: usize = 10;
const FINAL_NORM: usize = 11;

/// The rows a pass covers: `count` rows of the window at position `position`, whose attention starts at the key
/// `first_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// The position of the window's first row.
    pub position: u64,
    /// The first position whose depth-1 key and value the window attends.
    pub first_key: u64,
    /// The rows of the window, from 1 to the drafter's `rows`.
    pub count: u32,
}

/// The planned forward pass of a depth.
#[derive(Clone, Debug)]
pub struct Forward {
    /// The work, to enqueue in order.
    pub work: Vec<Work>,
    /// The depth's output hidden `[count, hidden]` (after the final norm), which the next depth takes as its input.
    pub output: u64,
}

/// The buffers of one depth: what its forward pass leaves for its backward pass, and its gradients.
#[derive(Clone, Copy, Debug)]
struct Depth {
    embedding_normed: u64,
    embedding_inverse: u64,
    hidden_normed: u64,
    hidden_inverse: u64,
    cat: u64,
    c: u64,
    n1: u64,
    inv1: u64,
    qk: u64,
    gv: u64,
    qkn: u64,
    qk_inverse: u64,
    probabilities: u64,
    extra_probabilities: u64,
    context: u64,
    gated: u64,
    h1: u64,
    n2: u64,
    inv2: u64,
    gate_up: u64,
    swiglu: u64,
    y: u64,
    m: u64,
    final_inverse: u64,
    d_m: u64,
    d_qkn: u64,
    d_gv: u64,
    /// The input hidden and embedded rows of the depth's last planned forward pass, for its backward pass.
    inputs: Option<(u64, u64)>,
}

/// The buffers one depth's backward pass uses and the next depth's reuses.
#[derive(Clone, Copy, Debug)]
struct Shared {
    d_y: u64,
    d_h1: u64,
    d_swiglu: u64,
    d_gate_up: u64,
    d_n2: u64,
    d_gated: u64,
    d_context: u64,
    d_scores: u64,
    d_extras: u64,
    d_qk: u64,
    d_n1: u64,
    d_c: u64,
    d_cat: u64,
    d_part: u64,
    d_discard: u64,
}

/// The work planned so far, with the products and kernels of the planners as its pieces.
#[derive(Default)]
struct Plan(Vec<Work>);

impl Plan {
    fn steps(&mut self, steps: Result<Vec<Step>, TrainError>) -> Result<(), DrafterError> {
        self.0.push(Work::Steps(steps?));
        Ok(())
    }

    fn product(&mut self, gemm: Result<Gemm, TrainError>) -> Result<(), DrafterError> {
        self.0.push(Work::Product(gemm?));
        Ok(())
    }
}

/// `address` moved on by `floats` floats.
fn at(address: u64, floats: u64) -> u64 {
    address + 4 * floats
}

/// The drafter model: the parameters and buffers of an unrolled draft over windows of rows, and the planners of its
/// passes.
///
/// Plain state of one owner: the parameters' addresses, the buffers' addresses and the inputs each depth's last forward
/// pass saw. The device memory is one arena that the caller allocates ([`arena_bytes`](Self::arena_bytes)) and keeps
/// alive while any planned work is in flight; the drafter only does arithmetic on its addresses.
///
/// Call sequence of a window: [`forward`](Self::forward) for depth 1 up to the deepest depth, each over the previous
/// output; [`start_backward`](Self::start_backward); the loss of each depth added into its
/// [`output_gradient`](Self::output_gradient); [`backward`](Self::backward) from the deepest depth down to depth 1.
/// Gradients accumulate across windows until [`step`](Self::step) applies and zeroes them.
#[derive(Debug)]
pub struct Drafter {
    config: Config,
    parameters: Vec<Parameter>,
    rounded: bool,
    weight: u64,
    keys: u64,
    values: u64,
    key_gradients: u64,
    value_gradients: u64,
    depth: Vec<Depth>,
    shared: Shared,
    bytes: u64,
}

impl Drafter {
    /// The bytes of device memory a drafter of `config` takes: its parameters, their gradients and moments, and its
    /// activations.
    pub fn arena_bytes(config: &Config) -> Result<u64, DrafterError> {
        Ok(Self::build(config, 0)?.bytes)
    }

    /// A drafter of `config` over the arena at `base`, which holds [`arena_bytes`](Self::arena_bytes) bytes, is 256-byte
    /// aligned and is zero where the gradients and moments start.
    pub fn new(config: &Config, base: u64) -> Result<Self, DrafterError> {
        if base == 0 || !base.is_multiple_of(256) {
            return Err(DrafterError::Arena(base));
        }
        Self::build(config, base)
    }

    fn build(config: &Config, base: u64) -> Result<Self, DrafterError> {
        config.validate()?;
        let mut arena = Arena::new(base);
        let parameters = layout::parameters(config, &mut arena);
        let shape = &config.shape;
        let rows = u64::from(config.rows);
        let (h, width, mixer) = (
            u64::from(shape.hidden),
            u64::from(shape.qk_width()),
            u64::from(shape.mixer_width()),
        );
        let (inter, heads) = (u64::from(shape.intermediate), u64::from(shape.heads));
        let stride = u64::from(config.depths.saturating_sub(1).max(1));
        let largest = parameters[..layout::PROJECTIONS.len()]
            .iter()
            .map(Parameter::count)
            .max()
            .unwrap_or(1);
        let weight = arena.take(largest);
        let kv = u64::from(config.max_positions) * u64::from(shape.kv_width());
        let (keys, values) = (arena.take(kv), arena.take(kv));
        let (key_gradients, value_gradients) = (arena.take(kv), arena.take(kv));
        let scores = heads * rows * u64::from(config.max_keys);
        let depth = (0..config.depths)
            .map(|_| Depth {
                embedding_normed: arena.take(rows * h),
                embedding_inverse: arena.take(rows),
                hidden_normed: arena.take(rows * h),
                hidden_inverse: arena.take(rows),
                cat: arena.take(rows * 2 * h),
                c: arena.take(rows * h),
                n1: arena.take(rows * h),
                inv1: arena.take(rows),
                qk: arena.take(rows * width),
                gv: arena.take(rows * width),
                qkn: arena.take(rows * width),
                qk_inverse: arena.take(rows * u64::from(shape.heads + shape.kv_heads)),
                probabilities: arena.take(scores),
                extra_probabilities: arena.take(heads * rows * stride),
                context: arena.take(rows * mixer),
                gated: arena.take(rows * mixer),
                h1: arena.take(rows * h),
                n2: arena.take(rows * h),
                inv2: arena.take(rows),
                gate_up: arena.take(rows * 2 * inter),
                swiglu: arena.take(rows * inter),
                y: arena.take(rows * h),
                m: arena.take(rows * h),
                final_inverse: arena.take(rows),
                d_m: arena.take(rows * h),
                d_qkn: arena.take(rows * width),
                d_gv: arena.take(rows * width),
                inputs: None,
            })
            .collect();
        let shared = Shared {
            d_y: arena.take(rows * h),
            d_h1: arena.take(rows * h),
            d_swiglu: arena.take(rows * inter),
            d_gate_up: arena.take(rows * 2 * inter),
            d_n2: arena.take(rows * h),
            d_gated: arena.take(rows * mixer),
            d_context: arena.take(rows * mixer),
            d_scores: arena.take(scores),
            d_extras: arena.take(heads * rows * stride),
            d_qk: arena.take(rows * width),
            d_n1: arena.take(rows * h),
            d_c: arena.take(rows * h),
            d_cat: arena.take(rows * 2 * h),
            d_part: arena.take(rows * h),
            d_discard: arena.take(rows * h),
        };
        Ok(Self {
            config: *config,
            parameters,
            rounded: true,
            weight,
            keys,
            values,
            key_gradients,
            value_gradients,
            depth,
            shared,
            bytes: arena.bytes(),
        })
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The parameters in parameter order: the five projections, then the seven norms.
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// The parameter named `name`.
    pub fn parameter(&self, name: &str) -> Option<&Parameter> {
        self.parameters.iter().find(|parameter| parameter.name == name)
    }

    /// Whether the projections run NVFP4-rounded (training, and evaluation of the exported layer) or as their FP32
    /// values. Planned passes keep the setting they were planned with.
    pub fn rounded(&self) -> bool {
        self.rounded
    }

    /// Sets whether the projections of later plans run NVFP4-rounded.
    pub fn set_rounded(&mut self, rounded: bool) {
        self.rounded = rounded;
    }

    /// The bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Recomputes each projection's NVFP4 global-scale source after its values changed.
    pub fn measure(&self) -> Result<Vec<Work>, DrafterError> {
        let mut plan = Plan::default();
        self.plan_measure(&mut plan)?;
        Ok(plan.0)
    }

    fn plan_measure(&self, plan: &mut Plan) -> Result<(), DrafterError> {
        for parameter in &self.parameters[..layout::PROJECTIONS.len()] {
            let Some(maximum) = parameter.maximum else {
                continue;
            };
            plan.steps(plan_zero(maximum, 1))?;
            plan.steps(plan_absolute_maximum(parameter.values, parameter.count(), maximum))?;
        }
        Ok(())
    }

    fn value(&self, index: usize) -> u64 {
        self.parameters[index].values
    }

    fn gradient(&self, index: usize) -> u64 {
        self.parameters[index].gradient
    }

    /// The projection's weight as the passes use it: NVFP4-rounded into the shared buffer, or its values.
    fn weight(&self, plan: &mut Plan, index: usize) -> Result<u64, DrafterError> {
        let parameter = &self.parameters[index];
        if !self.rounded {
            return Ok(parameter.values);
        }
        let maximum = parameter.maximum.unwrap_or(0);
        plan.steps(plan_nvfp4_round(
            parameter.values,
            self.weight,
            None,
            parameter.rows,
            parameter.columns,
            maximum,
        ))?;
        Ok(self.weight)
    }

    /// `dw [out, in] += dy^T x` into rows `[first_row, first_row + out)` of the projection's gradient.
    fn weight_gradient(
        &self,
        plan: &mut Plan,
        index: usize,
        (first_row, out): (u32, u32),
        dy: u64,
        x: u64,
        count: u32,
    ) -> Result<(), DrafterError> {
        let parameter = &self.parameters[index];
        let dw = at(parameter.gradient, u64::from(first_row) * u64::from(parameter.columns));
        let matrices = Matrices {
            a: (dy, out),
            b: (x, parameter.columns),
            c: (dw, parameter.columns),
        };
        plan.product(plan_product_left_transposed(
            &matrices,
            out,
            parameter.columns,
            count,
            1.0,
            1.0,
        ))
    }

    /// The keys a window attends, once its depth and rows are valid.
    fn check(&self, depth: u32, window: &Window) -> Result<u32, DrafterError> {
        let config = &self.config;
        if depth == 0 || depth > config.depths {
            return Err(DrafterError::Depth {
                depth,
                depths: config.depths,
            });
        }
        if window.count == 0 || window.count > config.rows {
            return Err(DrafterError::Rows {
                count: window.count,
                rows: config.rows,
            });
        }
        let end = window.position + u64::from(window.count);
        if window.first_key > window.position {
            return Err(DrafterError::FirstKey {
                first_key: window.first_key,
                position: window.position,
            });
        }
        let keys = end - window.first_key;
        if keys > u64::from(config.max_keys) {
            return Err(DrafterError::Keys {
                keys,
                max: config.max_keys,
            });
        }
        if end > u64::from(config.max_positions) {
            return Err(DrafterError::Position {
                position: end,
                max: config.max_positions,
            });
        }
        Ok(keys as u32)
    }

    /// The scale of an attention score: one over the root of the head width.
    fn score_scale(&self) -> f32 {
        (1.0 / f64::from(self.config.shape.head_dim).sqrt()) as f32
    }

    /// The rotation's head layout at the position of depth `depth` of `window`.
    fn heads(&self, depth: u32, window: &Window) -> Heads {
        let shape = &self.config.shape;
        Heads {
            rows: window.count,
            q_heads: shape.heads,
            kv_heads: shape.kv_heads,
            head: shape.head_dim,
            rotary: shape.rotary_dim,
            position0: (window.position + u64::from(depth) - 1) as i64,
            theta: shape.theta,
        }
    }

    /// The lineage kernels' geometry at `slot`.
    fn lineage(&self, window: &Window, slot: u32) -> Lineage {
        let shape = &self.config.shape;
        Lineage {
            heads: shape.heads,
            group: shape.heads / shape.kv_heads,
            d: shape.head_dim,
            rows: window.count,
            stride: self.extra_stride(),
            slot,
        }
    }

    /// The row length of the lineage tensors: one slot per depth after the first, and at least one.
    fn extra_stride(&self) -> u32 {
        self.config.depths.saturating_sub(1).max(1)
    }

    /// The softmax of depth `depth` over `window` and its `keys` causal keys, on the buffers `scores` and `extras`.
    fn softmax(&self, depth: u32, window: &Window, keys: u32, scores: u64, extras: u64) -> Softmax {
        Softmax {
            scores,
            ld_scores: keys,
            visible0: (window.position - window.first_key + 1) as u32,
            extras,
            extra_stride: self.extra_stride(),
            extra_count: depth - 1,
            heads: self.config.shape.heads,
            rows: window.count,
            scale: self.score_scale(),
        }
    }

    /// Plans depth `depth` (from 1) over `window`: `hidden` `[count, H]` is the depth's input hidden (the base model's
    /// for depth 1, the previous depth's output otherwise) and `embedded` `[count, H]` the embedding of each row's input
    /// token. Remembers both addresses for the backward pass. The depth-1 keys and values of the window are written
    /// into the sequence's keys and values at `position`.
    pub fn forward(
        &mut self,
        depth: u32,
        window: &Window,
        hidden: u64,
        embedded: u64,
    ) -> Result<Forward, DrafterError> {
        let keys = self.check(depth, window)?;
        let work = self.plan_forward(depth, window, keys, hidden, embedded)?;
        let buffers = &mut self.depth[depth as usize - 1];
        buffers.inputs = Some((hidden, embedded));
        Ok(Forward {
            work,
            output: buffers.m,
        })
    }

    fn plan_forward(
        &self,
        depth: u32,
        window: &Window,
        key_count: u32,
        hidden: u64,
        embedded: u64,
    ) -> Result<Vec<Work>, DrafterError> {
        let shape = self.config.shape;
        let (h, d, width, mixer, kvw) = (
            shape.hidden,
            shape.head_dim,
            shape.qk_width(),
            shape.mixer_width(),
            shape.kv_width(),
        );
        let inter = shape.intermediate;
        let group = shape.heads / shape.kv_heads;
        let count = window.count;
        let b = self.depth[depth as usize - 1];
        let mut plan = Plan::default();
        plan.steps(plan_rms_norm_forward(
            embedded,
            self.value(EMBEDDING_NORM),
            b.embedding_normed,
            b.embedding_inverse,
            count,
            h,
            shape.eps,
            NORM_OFFSET,
        ))?;
        plan.steps(plan_rms_norm_forward(
            hidden,
            self.value(HIDDEN_NORM),
            b.hidden_normed,
            b.hidden_inverse,
            count,
            h,
            shape.eps,
            NORM_OFFSET,
        ))?;
        plan.steps(plan_copy_columns(
            (b.embedding_normed, h),
            (b.cat, 2 * h),
            count,
            h,
            false,
        ))?;
        plan.steps(plan_copy_columns(
            (b.hidden_normed, h),
            (at(b.cat, u64::from(h)), 2 * h),
            count,
            h,
            false,
        ))?;
        let input_projection = self.weight(&mut plan, INPUT_PROJECTION)?;
        plan.product(plan_linear(b.cat, input_projection, b.c, count, 2 * h, h, false))?;
        plan.steps(plan_rms_norm_forward(
            b.c,
            self.value(INPUT_NORM),
            b.n1,
            b.inv1,
            count,
            h,
            shape.eps,
            NORM_OFFSET,
        ))?;
        let w = self.weight(&mut plan, QUERY_KEY_GATE_VALUE)?;
        plan.product(plan_linear(b.n1, w, b.qk, count, h, width, false))?;
        plan.product(plan_linear(
            b.n1,
            at(w, u64::from(width) * u64::from(h)),
            b.gv,
            count,
            h,
            width,
            false,
        ))?;
        plan.steps(plan_qk_norm_rope_forward(
            b.qk,
            self.value(QUERY_NORM),
            self.value(KEY_NORM),
            b.qkn,
            b.qk_inverse,
            &self.heads(depth, window),
            shape.eps,
        ))?;
        let first = window.first_key * u64::from(kvw);
        if depth == 1 {
            // The window's own depth-1 keys and values, which every depth's causal part reads.
            let at_position = window.position * u64::from(kvw);
            plan.steps(plan_copy_columns(
                (at(b.qkn, u64::from(mixer)), width),
                (at(self.keys, at_position), kvw),
                count,
                kvw,
                false,
            ))?;
            plan.steps(plan_copy_columns(
                (at(b.gv, u64::from(mixer)), width),
                (at(self.values, at_position), kvw),
                count,
                kvw,
                false,
            ))?;
        }
        let per_head = u64::from(count) * u64::from(key_count);
        for head in 0..shape.heads {
            let kv = u64::from(head / group);
            let matrices = Matrices {
                a: (at(b.qkn, u64::from(head) * u64::from(d)), width),
                b: (at(self.keys, first + kv * u64::from(d)), kvw),
                c: (at(b.probabilities, u64::from(head) * per_head), key_count),
            };
            plan.product(plan_product_transposed(&matrices, count, key_count, d, 1.0, 0.0))?;
        }
        for j in 2..=depth {
            let lineage = self.lineage(window, j - 2);
            let keys_of_j = self.depth[j as usize - 1].qkn;
            plan.steps(plan_lineage_dot(
                (b.qkn, width),
                (at(keys_of_j, u64::from(mixer)), width),
                b.extra_probabilities,
                &lineage,
            ))?;
        }
        let softmax = self.softmax(depth, window, key_count, b.probabilities, b.extra_probabilities);
        plan.steps(plan_softmax(&softmax))?;
        for head in 0..shape.heads {
            let kv = u64::from(head / group);
            let matrices = Matrices {
                a: (at(b.probabilities, u64::from(head) * per_head), key_count),
                b: (at(self.values, first + kv * u64::from(d)), kvw),
                c: (at(b.context, u64::from(head) * u64::from(d)), mixer),
            };
            plan.product(plan_product(&matrices, count, d, key_count, 1.0, 0.0))?;
        }
        for j in 2..=depth {
            let lineage = self.lineage(window, j - 2);
            let values_of_j = self.depth[j as usize - 1].gv;
            plan.steps(plan_lineage_spread(
                b.extra_probabilities,
                (at(values_of_j, u64::from(mixer)), width),
                (b.context, mixer),
                &lineage,
            ))?;
        }
        plan.steps(plan_gate_forward(b.context, b.gv, b.gated, count, mixer, width))?;
        plan.steps(plan_copy(b.c, b.h1, u64::from(count) * u64::from(h)))?;
        let output = self.weight(&mut plan, OUTPUT)?;
        plan.product(plan_linear(b.gated, output, b.h1, count, mixer, h, true))?;
        plan.steps(plan_rms_norm_forward(
            b.h1,
            self.value(POST_ATTENTION_NORM),
            b.n2,
            b.inv2,
            count,
            h,
            shape.eps,
            NORM_OFFSET,
        ))?;
        let gate_up = self.weight(&mut plan, GATE_UP)?;
        plan.product(plan_linear(b.n2, gate_up, b.gate_up, count, h, 2 * inter, false))?;
        plan.steps(plan_swiglu_forward(b.gate_up, b.swiglu, count, inter))?;
        plan.steps(plan_copy(b.h1, b.y, u64::from(count) * u64::from(h)))?;
        let down = self.weight(&mut plan, DOWN)?;
        plan.product(plan_linear(b.swiglu, down, b.y, count, inter, h, true))?;
        plan.steps(plan_rms_norm_forward(
            b.y,
            self.value(FINAL_NORM),
            b.m,
            b.final_inverse,
            count,
            h,
            shape.eps,
            NORM_OFFSET,
        ))?;
        Ok(plan.0)
    }

    /// The gradient buffer `[count, H]` of depth `depth`'s output hidden, which its loss and the next depth's backward
    /// pass add into. `None` for a depth outside `1..=depths`.
    pub fn output_gradient(&self, depth: u32) -> Option<u64> {
        let index = usize::try_from(depth).ok()?.checked_sub(1)?;
        self.depth.get(index).map(|buffers| buffers.d_m)
    }

    /// Zeroes what a window's backward passes accumulate across depths: every depth's output, query-key and gate-value
    /// gradients, and the window's key and value gradients.
    pub fn start_backward(&self, window: &Window) -> Result<Vec<Work>, DrafterError> {
        let keys = self.check(1, window)?;
        let shape = &self.config.shape;
        let (rows, kv_width) = (u64::from(window.count), u64::from(shape.kv_width()));
        let mut plan = Plan::default();
        for buffers in &self.depth {
            plan.steps(plan_zero(buffers.d_m, rows * u64::from(shape.hidden)))?;
            plan.steps(plan_zero(buffers.d_qkn, rows * u64::from(shape.qk_width())))?;
            plan.steps(plan_zero(buffers.d_gv, rows * u64::from(shape.qk_width())))?;
        }
        let first = window.first_key * kv_width;
        plan.steps(plan_zero(at(self.key_gradients, first), u64::from(keys) * kv_width))?;
        plan.steps(plan_zero(at(self.value_gradients, first), u64::from(keys) * kv_width))?;
        Ok(plan.0)
    }

    /// Plans depth `depth`'s backward pass from its output gradient (complete once every deeper depth's backward ran):
    /// adds the parameters' gradients, the shallower depths' query-key and gate-value gradients, and its input
    /// hidden's gradient into depth `depth - 1`'s output gradient. Run from the deepest depth to depth 1, after the
    /// forward passes of all of them.
    pub fn backward(&self, depth: u32, window: &Window) -> Result<Vec<Work>, DrafterError> {
        let key_count = self.check(depth, window)?;
        for shallower in 1..=depth {
            if self.depth[shallower as usize - 1].inputs.is_none() {
                return Err(DrafterError::NotForwarded(shallower));
            }
        }
        self.plan_backward(depth, window, key_count)
    }

    fn plan_backward(&self, depth: u32, window: &Window, key_count: u32) -> Result<Vec<Work>, DrafterError> {
        let shape = self.config.shape;
        let (h, d, width, mixer, kvw) = (
            shape.hidden,
            shape.head_dim,
            shape.qk_width(),
            shape.mixer_width(),
            shape.kv_width(),
        );
        let inter = shape.intermediate;
        let group = shape.heads / shape.kv_heads;
        let count = window.count;
        let (rows, hidden_floats) = (u64::from(count), u64::from(count) * u64::from(h));
        let b = self.depth[depth as usize - 1];
        let s = self.shared;
        let first = window.first_key * u64::from(kvw);
        let per_head = rows * u64::from(key_count);
        let mut plan = Plan::default();
        // The final norm.
        plan.steps(plan_zero(s.d_y, hidden_floats))?;
        plan.steps(plan_rms_norm_backward(
            b.y,
            self.value(FINAL_NORM),
            b.final_inverse,
            b.d_m,
            s.d_y,
            Some(self.gradient(FINAL_NORM)),
            count,
            h,
            NORM_OFFSET,
        ))?;
        // The feed-forward.
        plan.steps(plan_copy(s.d_y, s.d_h1, hidden_floats))?;
        plan.steps(plan_zero(s.d_swiglu, rows * u64::from(inter)))?;
        let down = self.weight(&mut plan, DOWN)?;
        plan.product(plan_linear_input_gradient(s.d_y, down, s.d_swiglu, count, inter, h))?;
        self.weight_gradient(&mut plan, DOWN, (0, h), s.d_y, b.swiglu, count)?;
        plan.steps(plan_zero(s.d_gate_up, rows * 2 * u64::from(inter)))?;
        plan.steps(plan_swiglu_backward(b.gate_up, s.d_swiglu, s.d_gate_up, count, inter))?;
        plan.steps(plan_zero(s.d_n2, hidden_floats))?;
        let gate_up = self.weight(&mut plan, GATE_UP)?;
        plan.product(plan_linear_input_gradient(
            s.d_gate_up,
            gate_up,
            s.d_n2,
            count,
            h,
            2 * inter,
        ))?;
        self.weight_gradient(&mut plan, GATE_UP, (0, 2 * inter), s.d_gate_up, b.n2, count)?;
        plan.steps(plan_rms_norm_backward(
            b.h1,
            self.value(POST_ATTENTION_NORM),
            b.inv2,
            s.d_n2,
            s.d_h1,
            Some(self.gradient(POST_ATTENTION_NORM)),
            count,
            h,
            NORM_OFFSET,
        ))?;
        // The attention's output and gate.
        plan.steps(plan_zero(s.d_gated, rows * u64::from(mixer)))?;
        let output = self.weight(&mut plan, OUTPUT)?;
        plan.product(plan_linear_input_gradient(s.d_h1, output, s.d_gated, count, mixer, h))?;
        self.weight_gradient(&mut plan, OUTPUT, (0, h), s.d_h1, b.gated, count)?;
        plan.steps(plan_zero(s.d_context, rows * u64::from(mixer)))?;
        plan.steps(plan_gate_backward(
            b.context,
            b.gv,
            s.d_gated,
            s.d_context,
            b.d_gv,
            count,
            mixer,
            width,
        ))?;
        // Softmax attention: dP = dC V^T (causal and lineage), dV += P^T dC.
        for head in 0..shape.heads {
            let kv = u64::from(head / group);
            let head_context = at(s.d_context, u64::from(head) * u64::from(d));
            let transposed = Matrices {
                a: (head_context, mixer),
                b: (at(self.values, first + kv * u64::from(d)), kvw),
                c: (at(s.d_scores, u64::from(head) * per_head), key_count),
            };
            plan.product(plan_product_transposed(&transposed, count, key_count, d, 1.0, 0.0))?;
            let left = Matrices {
                a: (at(b.probabilities, u64::from(head) * per_head), key_count),
                b: (head_context, mixer),
                c: (at(self.value_gradients, first + kv * u64::from(d)), kvw),
            };
            plan.product(plan_product_left_transposed(&left, key_count, d, count, 1.0, 1.0))?;
        }
        for j in 2..=depth {
            let lineage = self.lineage(window, j - 2);
            let of_j = self.depth[j as usize - 1];
            plan.steps(plan_lineage_dot(
                (s.d_context, mixer),
                (at(of_j.gv, u64::from(mixer)), width),
                s.d_extras,
                &lineage,
            ))?;
            plan.steps(plan_lineage_gather(
                b.extra_probabilities,
                (s.d_context, mixer),
                (at(of_j.d_gv, u64::from(mixer)), width),
                &lineage,
            ))?;
        }
        let softmax = self.softmax(depth, window, key_count, b.probabilities, b.extra_probabilities);
        plan.steps(plan_softmax_backward(&softmax, s.d_scores, s.d_extras))?;
        // dQ += dS K, dK += dS^T Q (causal and lineage).
        for head in 0..shape.heads {
            let kv = u64::from(head / group);
            let d_scores = at(s.d_scores, u64::from(head) * per_head);
            let queries = Matrices {
                a: (d_scores, key_count),
                b: (at(self.keys, first + kv * u64::from(d)), kvw),
                c: (at(b.d_qkn, u64::from(head) * u64::from(d)), width),
            };
            plan.product(plan_product(&queries, count, d, key_count, 1.0, 1.0))?;
            let keys = Matrices {
                a: (d_scores, key_count),
                b: (at(b.qkn, u64::from(head) * u64::from(d)), width),
                c: (at(self.key_gradients, first + kv * u64::from(d)), kvw),
            };
            plan.product(plan_product_left_transposed(&keys, key_count, d, count, 1.0, 1.0))?;
        }
        for j in 2..=depth {
            let lineage = self.lineage(window, j - 2);
            let of_j = self.depth[j as usize - 1];
            plan.steps(plan_lineage_spread(
                s.d_extras,
                (at(of_j.qkn, u64::from(mixer)), width),
                (b.d_qkn, width),
                &lineage,
            ))?;
            plan.steps(plan_lineage_gather(
                s.d_extras,
                (b.qkn, width),
                (at(of_j.d_qkn, u64::from(mixer)), width),
                &lineage,
            ))?;
        }
        if depth == 1 {
            // The window's own depth-1 keys and values: every depth's causal part read them.
            let at_position = window.position * u64::from(kvw);
            plan.steps(plan_copy_columns(
                (at(self.key_gradients, at_position), kvw),
                (at(b.d_qkn, u64::from(mixer)), width),
                count,
                kvw,
                true,
            ))?;
            plan.steps(plan_copy_columns(
                (at(self.value_gradients, at_position), kvw),
                (at(b.d_gv, u64::from(mixer)), width),
                count,
                kvw,
                true,
            ))?;
        }
        plan.steps(plan_zero(s.d_qk, rows * u64::from(width)))?;
        plan.steps(plan_qk_norm_rope_backward(
            b.qk,
            self.value(QUERY_NORM),
            self.value(KEY_NORM),
            b.qk_inverse,
            b.d_qkn,
            s.d_qk,
            Some(self.gradient(QUERY_NORM)),
            Some(self.gradient(KEY_NORM)),
            &self.heads(depth, window),
        ))?;
        plan.steps(plan_zero(s.d_n1, hidden_floats))?;
        let w = self.weight(&mut plan, QUERY_KEY_GATE_VALUE)?;
        let gate_value = at(w, u64::from(width) * u64::from(h));
        plan.product(plan_linear_input_gradient(s.d_qk, w, s.d_n1, count, h, width))?;
        plan.product(plan_linear_input_gradient(b.d_gv, gate_value, s.d_n1, count, h, width))?;
        self.weight_gradient(&mut plan, QUERY_KEY_GATE_VALUE, (0, width), s.d_qk, b.n1, count)?;
        self.weight_gradient(&mut plan, QUERY_KEY_GATE_VALUE, (width, width), b.d_gv, b.n1, count)?;
        plan.steps(plan_copy(s.d_h1, s.d_c, hidden_floats))?;
        plan.steps(plan_rms_norm_backward(
            b.c,
            self.value(INPUT_NORM),
            b.inv1,
            s.d_n1,
            s.d_c,
            Some(self.gradient(INPUT_NORM)),
            count,
            h,
            NORM_OFFSET,
        ))?;
        // The input projection and the two input norms.
        plan.steps(plan_zero(s.d_cat, rows * 2 * u64::from(h)))?;
        let input_projection = self.weight(&mut plan, INPUT_PROJECTION)?;
        plan.product(plan_linear_input_gradient(
            s.d_c,
            input_projection,
            s.d_cat,
            count,
            2 * h,
            h,
        ))?;
        self.weight_gradient(&mut plan, INPUT_PROJECTION, (0, h), s.d_c, b.cat, count)?;
        let (input_hidden, input_embedded) = self.depth[depth as usize - 1]
            .inputs
            .ok_or(DrafterError::NotForwarded(depth))?;
        plan.steps(plan_copy_columns((s.d_cat, 2 * h), (s.d_part, h), count, h, false))?;
        plan.steps(plan_rms_norm_backward(
            input_embedded,
            self.value(EMBEDDING_NORM),
            b.embedding_inverse,
            s.d_part,
            s.d_discard,
            Some(self.gradient(EMBEDDING_NORM)),
            count,
            h,
            NORM_OFFSET,
        ))?;
        plan.steps(plan_copy_columns(
            (at(s.d_cat, u64::from(h)), 2 * h),
            (s.d_part, h),
            count,
            h,
            false,
        ))?;
        let d_hidden = if depth > 1 {
            self.depth[depth as usize - 2].d_m
        } else {
            s.d_discard
        };
        plan.steps(plan_rms_norm_backward(
            input_hidden,
            self.value(HIDDEN_NORM),
            b.hidden_inverse,
            s.d_part,
            d_hidden,
            Some(self.gradient(HIDDEN_NORM)),
            count,
            h,
            NORM_OFFSET,
        ))?;
        Ok(plan.0)
    }

    /// One AdamW step (numbered from 1) of every parameter, which zeroes the gradients, then the projections' global-scale
    /// sources.
    pub fn step(&self, projection_rate: f32, norm_rate: f32, step: u32) -> Result<Vec<Work>, DrafterError> {
        let mut plan = Plan::default();
        for parameter in &self.parameters {
            let rate = if parameter.maximum.is_some() {
                projection_rate
            } else {
                norm_rate
            };
            let settings = AdamSettings {
                rate,
                beta1: FIRST_DECAY,
                beta2: SECOND_DECAY,
                eps: ADAM_EPS,
                decay: 0.0,
            };
            plan.steps(plan_adam(
                parameter.values,
                parameter.gradient,
                parameter.first,
                parameter.second,
                parameter.count(),
                &settings,
                step,
            ))?;
        }
        self.plan_measure(&mut plan)?;
        Ok(plan.0)
    }

    /// The shape of the layer.
    pub fn shape(&self) -> &Shape {
        &self.config.shape
    }
}
