//! The kernel planners: each function turns a shape and device addresses into the step one kernel runs.

use euhedral_gpu::{Args, Dim3, Launch};

use super::{Step, TrainError};
use crate::kernels::Module;

/// Threads of an elementwise or row-wise launch.
const THREADS: u32 = 256;
/// The widest head the rotary kernels hold in shared memory.
const MOST_HEAD: u32 = 512;

fn kernel_name(name: &str) -> Result<&'static str, TrainError> {
    Module::Train
        .kernels()
        .iter()
        .copied()
        .find(|listed| *listed == name)
        .ok_or_else(|| TrainError::NoKernel(name.to_owned()))
}

fn step(kernel: &str, blocks: u32, threads: u32, args: Args) -> Result<Step, TrainError> {
    Ok(Step {
        kernel: kernel_name(kernel)?,
        launch: Launch {
            grid: Dim3::x(blocks),
            block: Dim3::x(threads),
            shared_bytes: 0,
            programmatic: false,
        },
        args,
    })
}

pub(super) fn address(name: &'static str, value: u64) -> Result<u64, TrainError> {
    if value == 0 {
        Err(TrainError::NullAddress(name))
    } else {
        Ok(value)
    }
}

/// An address that may be absent: null stays null.
fn optional(value: Option<u64>) -> u64 {
    value.unwrap_or(0)
}

/// `value` as the kernel's positive signed 32-bit parameter.
pub(super) fn dimension(name: &'static str, value: u32) -> Result<i32, TrainError> {
    match i32::try_from(value) {
        Ok(signed) if signed > 0 => Ok(signed),
        _ => Err(TrainError::OutOfRange {
            name,
            value: i64::from(value),
        }),
    }
}

/// `value` as the kernel's positive signed 64-bit count.
fn count64(name: &'static str, value: u64) -> Result<i64, TrainError> {
    match i64::try_from(value) {
        Ok(signed) if signed > 0 => Ok(signed),
        _ => Err(TrainError::OutOfRange { name, value: -1 }),
    }
}

/// Blocks of an elementwise launch over `count` values.
fn elementwise(name: &'static str, count: u64) -> Result<u32, TrainError> {
    let blocks = count.div_ceil(u64::from(THREADS));
    match i32::try_from(blocks) {
        Ok(_) => Ok(blocks as u32),
        Err(_) => Err(TrainError::OutOfRange { name, value: -1 }),
    }
}

/// Blocks of a launch with one block per `rows * per_row`.
fn blocks(name: &'static str, rows: u32, per_row: u32) -> Result<u32, TrainError> {
    let total = u64::from(rows) * u64::from(per_row);
    match i32::try_from(total) {
        Ok(_) => Ok(total as u32),
        Err(_) => Err(TrainError::OutOfRange { name, value: -1 }),
    }
}

/// The elements `rows * width` of a launch, both being positive 32-bit values.
fn cells(rows: u32, width: u32) -> Result<u64, TrainError> {
    dimension("rows", rows)?;
    dimension("width", width)?;
    Ok(u64::from(rows) * u64::from(width))
}

/// Widens the BF16 values `[count]` at `input` to FP32 at `out`.
pub fn plan_bf16_to_float(input: u64, count: u64, out: u64) -> Result<Vec<Step>, TrainError> {
    let (input, out) = (address("input", input)?, address("out", out)?);
    let signed = count64("count", count)?;
    let args = Args::new().device(input).arg(signed).device(out);
    Ok(vec![step(
        "bf16_to_float",
        elementwise("count", count)?,
        THREADS,
        args,
    )?])
}

/// `y = x` for `count` floats.
pub fn plan_copy(x: u64, y: u64, count: u64) -> Result<Vec<Step>, TrainError> {
    let (x, y) = (address("x", x)?, address("y", y)?);
    let signed = count64("count", count)?;
    let args = Args::new().device(x).device(y).arg(signed);
    Ok(vec![step("copy_floats", elementwise("count", count)?, THREADS, args)?])
}

/// Sets `count` floats at `x` to `value`.
pub fn plan_fill(x: u64, count: u64, value: f32) -> Result<Vec<Step>, TrainError> {
    let x = address("x", x)?;
    let signed = count64("count", count)?;
    let args = Args::new().device(x).arg(signed).arg(value);
    Ok(vec![step("fill", elementwise("count", count)?, THREADS, args)?])
}

/// Sets `count` floats at `x` to zero.
pub fn plan_zero(x: u64, count: u64) -> Result<Vec<Step>, TrainError> {
    plan_fill(x, count, 0.0)
}

/// `h[r, j] = silu(gate[r, j]) * up[r, j]` for rows `[gate (width); up (width)]` of `gate_up`, into `h` `[rows, width]`.
pub fn plan_swiglu_forward(gate_up: u64, h: u64, rows: u32, width: u32) -> Result<Vec<Step>, TrainError> {
    let (gate_up, h) = (address("gate_up", gate_up)?, address("h", h)?);
    let total = cells(rows, width)?;
    let args = Args::new()
        .device(gate_up)
        .device(h)
        .arg(dimension("rows", rows)?)
        .arg(dimension("width", width)?);
    Ok(vec![step(
        "swiglu_forward",
        elementwise("count", total)?,
        THREADS,
        args,
    )?])
}

/// Adds the gradient of [`plan_swiglu_forward`] to `d_gate_up` `[rows, 2 * width]` from `dh` `[rows, width]`.
pub fn plan_swiglu_backward(
    gate_up: u64,
    dh: u64,
    d_gate_up: u64,
    rows: u32,
    width: u32,
) -> Result<Vec<Step>, TrainError> {
    let (gate_up, dh) = (address("gate_up", gate_up)?, address("dh", dh)?);
    let d_gate_up = address("d_gate_up", d_gate_up)?;
    let total = cells(rows, width)?;
    let args = Args::new()
        .device(gate_up)
        .device(dh)
        .device(d_gate_up)
        .arg(dimension("rows", rows)?)
        .arg(dimension("width", width)?);
    Ok(vec![step(
        "swiglu_backward",
        elementwise("count", total)?,
        THREADS,
        args,
    )?])
}

/// `y = x * inv * (offset + w)` per row of `width`, with `inv = rsqrt(mean(x^2) + eps)` kept at `inverse` `[rows]`.
#[allow(clippy::too_many_arguments, reason = "the kernel's own parameter list")]
pub fn plan_rms_norm_forward(
    x: u64,
    w: u64,
    y: u64,
    inverse: u64,
    rows: u32,
    width: u32,
    eps: f32,
    offset: f32,
) -> Result<Vec<Step>, TrainError> {
    let (x, w) = (address("x", x)?, address("w", w)?);
    let (y, inverse) = (address("y", y)?, address("inverse", inverse)?);
    dimension("rows", rows)?;
    let args = Args::new()
        .device(x)
        .device(w)
        .device(y)
        .device(inverse)
        .arg(dimension("width", width)?)
        .arg(eps)
        .arg(offset);
    Ok(vec![step("rms_norm_forward", rows, THREADS, args)?])
}

/// Adds the gradient of [`plan_rms_norm_forward`] to `dx`, and to `dw` `[width]` when it is given. The weight
/// gradient sums over rows with atomic adds, so its last bits depend on the order the blocks run in.
#[allow(clippy::too_many_arguments, reason = "the kernel's own parameter list")]
pub fn plan_rms_norm_backward(
    x: u64,
    w: u64,
    inverse: u64,
    dy: u64,
    dx: u64,
    dw: Option<u64>,
    rows: u32,
    width: u32,
    offset: f32,
) -> Result<Vec<Step>, TrainError> {
    let (x, w, inverse) = (address("x", x)?, address("w", w)?, address("inverse", inverse)?);
    let (dy, dx) = (address("dy", dy)?, address("dx", dx)?);
    dimension("rows", rows)?;
    let args = Args::new()
        .device(x)
        .device(w)
        .device(inverse)
        .device(dy)
        .device(dx)
        .device(optional(dw))
        .arg(dimension("width", width)?)
        .arg(offset);
    Ok(vec![step("rms_norm_backward", rows, THREADS, args)?])
}

/// The head layout and rotation of the query and key norm with RoPE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heads {
    /// Rows (tokens) of the input.
    pub rows: u32,
    /// Query heads per row; the key heads follow them.
    pub q_heads: u32,
    /// Key heads per row.
    pub kv_heads: u32,
    /// The width of a head.
    pub head: u32,
    /// The leading dimensions that rotate, in halves.
    pub rotary: u32,
    /// The position of row zero.
    pub position0: i64,
    /// The rotation's base.
    pub theta: f64,
}

impl Heads {
    fn blocks(&self) -> Result<u32, TrainError> {
        dimension("rows", self.rows)?;
        dimension("q_heads", self.q_heads)?;
        let kv = i32::try_from(self.kv_heads).map_err(|_| TrainError::OutOfRange {
            name: "kv_heads",
            value: i64::from(self.kv_heads),
        })?;
        dimension("head", self.head)?;
        if !self.head.is_multiple_of(32) || self.head > MOST_HEAD {
            return Err(TrainError::HeadDimension(self.head));
        }
        if !self.rotary.is_multiple_of(2) || self.rotary > self.head {
            return Err(TrainError::Rotary {
                rotary: self.rotary,
                head: self.head,
            });
        }
        let heads = self.q_heads.checked_add(kv as u32).ok_or(TrainError::OutOfRange {
            name: "heads",
            value: -1,
        })?;
        if u64::from(heads) * u64::from(self.head) > i32::MAX as u64 {
            return Err(TrainError::OutOfRange {
                name: "heads",
                value: i64::from(heads),
            });
        }
        blocks("rows", self.rows, heads)
    }
}

/// Normalises each head of rows `[q (q_heads x head); k (kv_heads x head)]` by its own RMS with the weight `1 + w`
/// (`q_norm` for query heads, `k_norm` for key heads), then rotates the first `rotary` dimensions in halves at the
/// angle `(position0 + row) * theta^(-2 f / rotary)`. `inverse` keeps one value per row and head.
pub fn plan_qk_norm_rope_forward(
    qk: u64,
    q_norm: u64,
    k_norm: u64,
    out: u64,
    inverse: u64,
    shape: &Heads,
    eps: f32,
) -> Result<Vec<Step>, TrainError> {
    let (qk, q_norm, k_norm) = (
        address("qk", qk)?,
        address("q_norm", q_norm)?,
        address("k_norm", k_norm)?,
    );
    let (out, inverse) = (address("out", out)?, address("inverse", inverse)?);
    let blocks = shape.blocks()?;
    let args = Args::new()
        .device(qk)
        .device(q_norm)
        .device(k_norm)
        .device(out)
        .device(inverse)
        .arg(dimension("q_heads", shape.q_heads)?)
        .arg(shape.kv_heads as i32)
        .arg(dimension("head", shape.head)?)
        .arg(shape.rotary as i32)
        .arg(shape.position0)
        .arg(shape.theta)
        .arg(eps);
    Ok(vec![step("qk_norm_rope_forward", blocks, shape.head, args)?])
}

/// Adds the gradient of [`plan_qk_norm_rope_forward`] to `dqk`, and the weight gradients to `d_q_norm` and
/// `d_k_norm` (`[head]` each) when they are given. The weight gradients sum with atomic adds, so their last bits
/// depend on the order the blocks run in.
#[allow(clippy::too_many_arguments, reason = "the kernel's own parameter list")]
pub fn plan_qk_norm_rope_backward(
    qk: u64,
    q_norm: u64,
    k_norm: u64,
    inverse: u64,
    d_out: u64,
    dqk: u64,
    d_q_norm: Option<u64>,
    d_k_norm: Option<u64>,
    shape: &Heads,
) -> Result<Vec<Step>, TrainError> {
    let (qk, q_norm, k_norm) = (
        address("qk", qk)?,
        address("q_norm", q_norm)?,
        address("k_norm", k_norm)?,
    );
    let (inverse, d_out, dqk) = (
        address("inverse", inverse)?,
        address("d_out", d_out)?,
        address("dqk", dqk)?,
    );
    let blocks = shape.blocks()?;
    let args = Args::new()
        .device(qk)
        .device(q_norm)
        .device(k_norm)
        .device(inverse)
        .device(d_out)
        .device(dqk)
        .device(optional(d_q_norm))
        .device(optional(d_k_norm))
        .arg(dimension("q_heads", shape.q_heads)?)
        .arg(shape.kv_heads as i32)
        .arg(dimension("head", shape.head)?)
        .arg(shape.rotary as i32)
        .arg(shape.position0)
        .arg(shape.theta);
    Ok(vec![step("qk_norm_rope_backward", blocks, shape.head, args)?])
}

/// `out[r, c] = context[r, c] * sigmoid(gate[r, c])` where `gate` rows are `row_width` apart and the first `width`
/// of each is used.
pub fn plan_gate_forward(
    context: u64,
    gate: u64,
    out: u64,
    rows: u32,
    width: u32,
    row_width: u32,
) -> Result<Vec<Step>, TrainError> {
    let (context, gate, out) = (
        address("context", context)?,
        address("gate", gate)?,
        address("out", out)?,
    );
    let total = cells(rows, width)?;
    let args = Args::new()
        .device(context)
        .device(gate)
        .device(out)
        .arg(dimension("rows", rows)?)
        .arg(dimension("width", width)?)
        .arg(gate_row("row_width", width, row_width)?);
    Ok(vec![step("gate_forward", elementwise("count", total)?, THREADS, args)?])
}

/// Adds the gradient of [`plan_gate_forward`] to `d_context` and `d_gate`.
#[allow(clippy::too_many_arguments, reason = "the kernel's own parameter list")]
pub fn plan_gate_backward(
    context: u64,
    gate: u64,
    d_out: u64,
    d_context: u64,
    d_gate: u64,
    rows: u32,
    width: u32,
    row_width: u32,
) -> Result<Vec<Step>, TrainError> {
    let (context, gate) = (address("context", context)?, address("gate", gate)?);
    let (d_out, d_context, d_gate) = (
        address("d_out", d_out)?,
        address("d_context", d_context)?,
        address("d_gate", d_gate)?,
    );
    let total = cells(rows, width)?;
    let args = Args::new()
        .device(context)
        .device(gate)
        .device(d_out)
        .device(d_context)
        .device(d_gate)
        .arg(dimension("rows", rows)?)
        .arg(dimension("width", width)?)
        .arg(gate_row("row_width", width, row_width)?);
    Ok(vec![step(
        "gate_backward",
        elementwise("count", total)?,
        THREADS,
        args,
    )?])
}

/// `row_width` as a signed parameter, when a row holds the `width` used.
fn gate_row(name: &'static str, width: u32, row_width: u32) -> Result<i32, TrainError> {
    if row_width < width {
        return Err(TrainError::ShortLeading {
            name,
            leading: i32::try_from(row_width).unwrap_or(i32::MAX),
            row: i32::try_from(width).unwrap_or(i32::MAX),
        });
    }
    dimension(name, row_width)
}

/// The hyperparameters of an AdamW step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamSettings {
    /// The learning rate.
    pub rate: f32,
    /// The decay of the first moment.
    pub beta1: f32,
    /// The decay of the second moment.
    pub beta2: f32,
    /// Added to the second moment's root.
    pub eps: f32,
    /// The weight decay.
    pub decay: f32,
}

/// One AdamW step at `step` (from 1, for the bias correction) over `count` parameters `p` with gradient `g` and
/// moments `m` and `v`. The gradient is zeroed.
pub fn plan_adam(
    p: u64,
    g: u64,
    m: u64,
    v: u64,
    count: u64,
    settings: &AdamSettings,
    step_number: u32,
) -> Result<Vec<Step>, TrainError> {
    let (p, g, m, v) = (address("p", p)?, address("g", g)?, address("m", m)?, address("v", v)?);
    let signed = count64("count", count)?;
    let args = Args::new()
        .device(p)
        .device(g)
        .device(m)
        .device(v)
        .arg(signed)
        .arg(settings.rate)
        .arg(settings.beta1)
        .arg(settings.beta2)
        .arg(settings.eps)
        .arg(settings.decay)
        .arg(dimension("step", step_number)?);
    Ok(vec![step("adam", elementwise("count", count)?, THREADS, args)?])
}
