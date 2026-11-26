//! The planners of the drafter module's kernels: each turns a shape and device addresses into the step one kernel
//! runs. A planner touches no device and refuses every shape a launch would fault on.

use euhedral_gpu::{Args, Dim3, Launch};

use crate::kernels::{Module, Step};
use crate::training::{THREADS, TrainError, address, count64, dimension, elementwise, out_of_range};

/// The most blocks `absolute_maximum` runs: each strides over what is left.
const MOST_MAXIMUM_BLOCKS: u64 = 1024;

fn step(kernel: &str, blocks: u32, threads: u32, args: Args) -> Result<Step, TrainError> {
    let launch = Launch {
        grid: Dim3::x(blocks),
        block: Dim3::x(threads),
        shared_bytes: 0,
        programmatic: false,
    };
    Step::new(Module::Drafter, kernel, launch, args).ok_or_else(|| TrainError::NoKernel(kernel.to_owned()))
}

/// A leading dimension `leading` as a signed parameter, when a row of `width` fits in it.
fn row(name: &'static str, leading: u32, width: u32) -> Result<i32, TrainError> {
    let signed = dimension(name, leading)?;
    if leading < width {
        return Err(TrainError::ShortLeading {
            name,
            leading: signed,
            row: i32::try_from(width).unwrap_or(i32::MAX),
        });
    }
    Ok(signed)
}

/// `rows * per_row` blocks, which must fit the kernel's signed 32-bit index.
fn blocks(name: &'static str, rows: u32, per_row: u32) -> Result<u32, TrainError> {
    let total = u64::from(rows) * u64::from(per_row);
    match i32::try_from(total) {
        Ok(_) => Ok(total as u32),
        Err(_) => Err(out_of_range(name, total)),
    }
}

/// `out` (one unsigned on the device, zeroed first) becomes the bits of the largest `|x|` over `count` floats.
pub fn plan_absolute_maximum(x: u64, count: u64, out: u64) -> Result<Vec<Step>, TrainError> {
    let (x, out) = (address("x", x)?, address("out", out)?);
    let signed = count64("count", count)?;
    let grid = count.div_ceil(u64::from(THREADS)).min(MOST_MAXIMUM_BLOCKS);
    let args = Args::new().device(x).arg(signed).device(out);
    Ok(vec![step("absolute_maximum", grid as u32, THREADS, args)?])
}

/// The packed codes `[rows, k / 2]` and the scale codes `[rows, k / 16]` of the artifact layout, which
/// [`plan_nvfp4_round`] writes besides the rounded values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Codes {
    /// The packed 4-bit codes, low nibble first.
    pub packed: u64,
    /// The E4M3 scale code of each block of 16.
    pub scales: u64,
}

/// `out` = `w` `[rows, k]` rounded to NVFP4 under the global scale of `amax_bits` (from [`plan_absolute_maximum`]),
/// and the artifact's codes when `codes` is given. `k` is a multiple of 16; `out` may be `w`.
pub fn plan_nvfp4_round(
    w: u64,
    out: u64,
    codes: Option<Nvfp4Codes>,
    rows: u32,
    k: u32,
    amax_bits: u64,
) -> Result<Vec<Step>, TrainError> {
    let (w, out, amax_bits) = (address("w", w)?, address("out", out)?, address("amax_bits", amax_bits)?);
    dimension("rows", rows)?;
    dimension("k", k)?;
    if !k.is_multiple_of(16) {
        return Err(out_of_range("k", u64::from(k)));
    }
    let total = u64::from(rows) * u64::from(k / 16);
    let signed = count64("blocks", total)?;
    let (packed, scales) = codes.map_or((0, 0), |codes| (codes.packed, codes.scales));
    if codes.is_some() {
        address("packed", packed)?;
        address("scales", scales)?;
    }
    let args = Args::new()
        .device(w)
        .device(out)
        .device(packed)
        .device(scales)
        .arg(signed)
        .device(amax_bits);
    Ok(vec![step("nvfp4_round", elementwise("blocks", total)?, THREADS, args)?])
}

/// The causal and lineage parts of one attention softmax: `scores` `[heads, rows, ld_scores]` and `extras`
/// `[heads, rows, extra_stride]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softmax {
    /// The scores or their probabilities.
    pub scores: u64,
    /// The row length of the scores, the causal keys.
    pub ld_scores: u32,
    /// The causal keys visible to row zero; row `r` sees `visible0 + r`.
    pub visible0: u32,
    /// The lineage scores or their probabilities.
    pub extras: u64,
    /// The row length of the extras.
    pub extra_stride: u32,
    /// The lineage slots in use.
    pub extra_count: u32,
    /// The heads.
    pub heads: u32,
    /// The rows.
    pub rows: u32,
    /// The factor on a score before the softmax.
    pub scale: f32,
}

impl Softmax {
    fn check(&self) -> Result<u32, TrainError> {
        dimension("ld_scores", self.ld_scores)?;
        dimension("extra_stride", self.extra_stride)?;
        dimension("rows", self.rows)?;
        dimension("heads", self.heads)?;
        if self.extra_count > self.extra_stride {
            return Err(out_of_range("extra_count", u64::from(self.extra_count)));
        }
        if u64::from(self.visible0) + u64::from(self.rows) - 1 > u64::from(self.ld_scores) {
            return Err(out_of_range("visible0", u64::from(self.visible0)));
        }
        i32::try_from(self.extra_count).map_err(|_| out_of_range("extra_count", u64::from(self.extra_count)))?;
        i32::try_from(self.visible0).map_err(|_| out_of_range("visible0", u64::from(self.visible0)))?;
        blocks("heads", self.heads, self.rows)
    }
}

/// The softmax in place over each `(head, row)`: causal columns `[0, visible0 + r)` of the scores and the
/// `extra_count` lineage slots; masked causal columns become 0.
pub fn plan_softmax(softmax: &Softmax) -> Result<Vec<Step>, TrainError> {
    let (scores, extras) = (address("scores", softmax.scores)?, address("extras", softmax.extras)?);
    let grid = softmax.check()?;
    let args = Args::new()
        .device(scores)
        .arg(softmax.ld_scores as i32)
        .arg(softmax.visible0 as i32)
        .device(extras)
        .arg(softmax.extra_stride as i32)
        .arg(softmax.extra_count as i32)
        .arg(softmax.rows as i32)
        .arg(softmax.scale);
    Ok(vec![step("drafter_softmax", grid, THREADS, args)?])
}

/// The gradient of [`plan_softmax`]: `dS = P (dP - sum(P dP)) scale` in place of `dP` (`d_scores`, `d_extras`), given the
/// probabilities `softmax` names.
pub fn plan_softmax_backward(softmax: &Softmax, d_scores: u64, d_extras: u64) -> Result<Vec<Step>, TrainError> {
    let (probabilities, extras) = (address("scores", softmax.scores)?, address("extras", softmax.extras)?);
    let (d_scores, d_extras) = (address("d_scores", d_scores)?, address("d_extras", d_extras)?);
    let grid = softmax.check()?;
    let args = Args::new()
        .device(probabilities)
        .device(d_scores)
        .arg(softmax.ld_scores as i32)
        .arg(softmax.visible0 as i32)
        .device(extras)
        .device(d_extras)
        .arg(softmax.extra_stride as i32)
        .arg(softmax.extra_count as i32)
        .arg(softmax.rows as i32)
        .arg(softmax.scale);
    Ok(vec![step("drafter_softmax_backward", grid, THREADS, args)?])
}

/// The heads, grouping and slot of a lineage kernel: `heads` query heads over `heads / group` key heads of width `d`,
/// `rows` rows, the lineage tensor `[heads, rows, stride]` at `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lineage {
    /// The query heads.
    pub heads: u32,
    /// The query heads that share one key head.
    pub group: u32,
    /// The head width: a multiple of 32 up to 512.
    pub d: u32,
    /// The rows.
    pub rows: u32,
    /// The row length of the lineage tensor.
    pub stride: u32,
    /// The slot of the tensor this kernel reads or writes.
    pub slot: u32,
}

impl Lineage {
    /// The key heads, once the shape is valid.
    fn key_heads(&self) -> Result<u32, TrainError> {
        dimension("heads", self.heads)?;
        dimension("group", self.group)?;
        dimension("rows", self.rows)?;
        dimension("stride", self.stride)?;
        if !self.d.is_multiple_of(32) || self.d == 0 || self.d > 512 {
            return Err(TrainError::HeadDimension(self.d));
        }
        if !self.heads.is_multiple_of(self.group) {
            return Err(out_of_range("group", u64::from(self.group)));
        }
        if self.slot >= self.stride {
            return Err(out_of_range("slot", u64::from(self.slot)));
        }
        Ok(self.heads / self.group)
    }
}

/// `out[(h rows + r) stride + slot] = a[r, h d ..] . b[r, (h / group) d ..]`: a lineage score (`a` the queries, `b` one
/// depth's keys) or its gradient (`a` the context's gradient, `b` the values). `a` and `b` have rows `lda` and `ldb`
/// apart.
pub fn plan_lineage_dot(a: (u64, u32), b: (u64, u32), out: u64, lineage: &Lineage) -> Result<Vec<Step>, TrainError> {
    let (a_address, b_address, out) = (address("a", a.0)?, address("b", b.0)?, address("out", out)?);
    let grid = blocks("rows", lineage.rows, lineage.heads)?;
    let key_heads = lineage.key_heads()?;
    let args = Args::new()
        .device(a_address)
        .arg(row("lda", a.1, lineage.heads * lineage.d)?)
        .device(b_address)
        .arg(row("ldb", b.1, key_heads * lineage.d)?)
        .arg(lineage.heads as i32)
        .arg(lineage.group as i32)
        .arg(lineage.d as i32)
        .arg(lineage.rows as i32)
        .device(out)
        .arg(lineage.stride as i32)
        .arg(lineage.slot as i32);
    Ok(vec![step("lineage_dot", grid, lineage.d, args)?])
}

/// `y[r, h d + i] += c[(h rows + r) stride + slot] x[r, (h / group) d + i]`: the lineage part of the context (`x` the
/// values) or of the queries' gradient (`x` the keys).
pub fn plan_lineage_spread(c: u64, x: (u64, u32), y: (u64, u32), lineage: &Lineage) -> Result<Vec<Step>, TrainError> {
    let (c, x_address, y_address) = (address("c", c)?, address("x", x.0)?, address("y", y.0)?);
    let grid = blocks("rows", lineage.rows, lineage.heads)?;
    let key_heads = lineage.key_heads()?;
    let args = Args::new()
        .device(c)
        .arg(lineage.stride as i32)
        .arg(lineage.slot as i32)
        .device(x_address)
        .arg(row("ldx", x.1, key_heads * lineage.d)?)
        .device(y_address)
        .arg(row("ldy", y.1, lineage.heads * lineage.d)?)
        .arg(lineage.heads as i32)
        .arg(lineage.group as i32)
        .arg(lineage.d as i32)
        .arg(lineage.rows as i32);
    Ok(vec![step("lineage_spread", grid, lineage.d, args)?])
}

/// `y[r, kv d + i] += sum over the group of kv of c[(h rows + r) stride + slot] x[r, h d + i]`: the lineage keys' or
/// values' gradient (`x` the queries or the context's gradient).
pub fn plan_lineage_gather(c: u64, x: (u64, u32), y: (u64, u32), lineage: &Lineage) -> Result<Vec<Step>, TrainError> {
    let (c, x_address, y_address) = (address("c", c)?, address("x", x.0)?, address("y", y.0)?);
    let key_heads = lineage.key_heads()?;
    let grid = blocks("rows", lineage.rows, key_heads)?;
    let args = Args::new()
        .device(c)
        .arg(lineage.stride as i32)
        .arg(lineage.slot as i32)
        .device(x_address)
        .arg(row("ldx", x.1, lineage.heads * lineage.d)?)
        .device(y_address)
        .arg(row("ldy", y.1, key_heads * lineage.d)?)
        .arg(lineage.heads as i32)
        .arg(lineage.group as i32)
        .arg(lineage.d as i32)
        .arg(lineage.rows as i32);
    Ok(vec![step("lineage_gather", grid, lineage.d, args)?])
}

/// `y[r, c] = x[r, c]` (or `+=` when `add`) for `c < columns`, between row-major buffers with rows `ldx` and `ldy`
/// apart.
pub fn plan_copy_columns(
    x: (u64, u32),
    y: (u64, u32),
    rows: u32,
    columns: u32,
    add: bool,
) -> Result<Vec<Step>, TrainError> {
    let (x_address, y_address) = (address("x", x.0)?, address("y", y.0)?);
    dimension("rows", rows)?;
    dimension("columns", columns)?;
    let args = Args::new()
        .device(x_address)
        .arg(row("ldx", x.1, columns)?)
        .device(y_address)
        .arg(row("ldy", y.1, columns)?)
        .arg(rows as i32)
        .arg(columns as i32)
        .arg(i32::from(add));
    let total = u64::from(rows) * u64::from(columns);
    Ok(vec![step("copy_columns", elementwise("count", total)?, THREADS, args)?])
}

/// The draft head's cross-entropy over `logits` `[rows, vocabulary]`, in place. A row with a label (`>= 0` in `labels`
/// `[rows]`) becomes `(softmax - onehot(label)) scale` and adds `-log p(label)` to `loss` (one double on the device); a
/// row without one becomes 0. `predicted` `[rows]` gets each row's argmax and `label_log_probability` `[rows]` the
/// label's log-probability.
#[allow(clippy::too_many_arguments, reason = "the kernel's own parameter list")]
pub fn plan_cross_entropy(
    logits: u64,
    rows: u32,
    vocabulary: u32,
    labels: u64,
    scale: f32,
    loss: u64,
    predicted: u64,
    label_log_probability: u64,
) -> Result<Vec<Step>, TrainError> {
    let logits = address("logits", logits)?;
    let (labels, loss) = (address("labels", labels)?, address("loss", loss)?);
    let (predicted, label_log_probability) = (
        address("predicted", predicted)?,
        address("label_log_probability", label_log_probability)?,
    );
    dimension("rows", rows)?;
    let args = Args::new()
        .device(logits)
        .arg(dimension("vocabulary", vocabulary)?)
        .device(labels)
        .arg(scale)
        .device(loss)
        .device(predicted)
        .device(label_log_probability);
    Ok(vec![step("draft_cross_entropy", rows, THREADS, args)?])
}
