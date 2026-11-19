//! The launch planners of the quantizer's own kernels: the scale, factor and refit passes around the Viterbi encode.
//!
//! Each function names the grid its kernel is written for. The reductions stride by their block size, so a grid or a
//! block that differs from these changes the rounding of the sums.

use euhedral_gpu::{Args, Dim3};

use super::plan::{address, elementwise, grid_rows, int, long, multiple, step};
use super::{Step, TrellisError};

/// Blocks of `count` items at `per` to a block.
fn blocks(count: u32, per: u32) -> u32 {
    count.div_ceil(per).max(1)
}

/// `W [k, n]` in float32 from the BF16 `nn.Linear` weight `[n, k]` at `weight`.
pub(super) fn plan_weight_to_float(weight: u64, n: u32, k: u32, out: u64) -> Result<Step, TrellisError> {
    let (weight, out) = (address("weight", weight)?, address("out", out)?);
    let args = Args::new()
        .device(weight)
        .arg(int("n", n)?)
        .arg(int("k", k)?)
        .device(out);
    step(
        "bf16_to_float_transposed",
        Dim3::new(blocks(k, 32), blocks(n, 32), 1),
        256,
        args,
    )
}

/// The mean of `count` floats of `v` spaced `stride` apart, as one float at `out`. One block.
pub(super) fn plan_strided_mean(v: u64, count: u32, stride: u64, out: u64) -> Result<Step, TrellisError> {
    let (v, out) = (address("v", v)?, address("out", out)?);
    let args = Args::new()
        .device(v)
        .arg(int("count", count)?)
        .arg(long("stride", stride)?)
        .device(out);
    step("strided_mean", Dim3::x(1), 1024, args)
}

/// Copies `count` floats.
pub(super) fn plan_copy_floats(src: u64, dst: u64, count: u64) -> Result<Step, TrellisError> {
    let (src, dst) = (address("src", src)?, address("dst", dst)?);
    let args = Args::new().device(src).device(dst).arg(long("count", count)?);
    step("copy_floats", Dim3::x(elementwise(count)), 256, args)
}

/// Adds `factor * mean[0]` to each diagonal entry of the `k` by `k` matrix `h`.
pub(super) fn plan_add_diagonal(h: u64, k: u32, mean: u64, factor: f32) -> Result<Step, TrellisError> {
    let (h, mean) = (address("h", h)?, address("mean", mean)?);
    let args = Args::new().device(h).arg(int("k", k)?).device(mean).arg(factor);
    step("add_diagonal", Dim3::x(blocks(k, 256)), 256, args)
}

/// Stores the diagonal of `h` into `d` (`store`) or writes `d` onto the diagonal.
pub(super) fn plan_diagonal_io(h: u64, k: u32, d: u64, store: bool) -> Result<Step, TrellisError> {
    let (h, d) = (address("h", h)?, address("d", d)?);
    let args = Args::new().device(h).arg(int("k", k)?).device(d).arg(i32::from(store));
    step("diagonal_io", Dim3::x(blocks(k, 256)), 256, args)
}

/// What [`plan_triangle`] does to the `k` by `k` matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Triangle {
    /// Copies the row-major lower triangle onto the upper.
    LowerToUpper = 0,
    /// Copies the upper onto the lower.
    UpperToLower = 1,
    /// Zeroes the upper.
    ClearUpper = 2,
}

/// One of the triangle copies of `h`.
pub(super) fn plan_triangle(h: u64, k: u32, mode: Triangle) -> Result<Step, TrellisError> {
    let h = address("h", h)?;
    let args = Args::new().device(h).arg(int("k", k)?).arg(mode as i32);
    step(
        "triangle",
        Dim3::new(blocks(k, 32), grid_rows("k / 8", blocks(k, 8))?, 1),
        256,
        args,
    )
}

/// Whether [`plan_scale_floats`] multiplies or divides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scaling {
    Multiply = 0,
    Divide = 1,
}

/// Which index of `x [rows, cols]` picks the factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Axis {
    /// `s[row]`.
    Rows = 0,
    /// `s[column]`.
    Columns = 1,
}

/// `x[r, c] *= s[axis]` (or `/=`).
pub(super) fn plan_scale_floats(
    x: u64,
    rows: u32,
    cols: u32,
    s: u64,
    axis: Axis,
    scaling: Scaling,
) -> Result<Step, TrellisError> {
    let (x, s) = (address("x", x)?, address("s", s)?);
    let args = Args::new()
        .device(x)
        .arg(int("rows", rows)?)
        .arg(int("cols", cols)?)
        .device(s)
        .arg(axis as i32)
        .arg(scaling as i32);
    step(
        "scale_floats",
        Dim3::new(blocks(cols, 256), grid_rows("rows", rows.max(1))?, 1),
        256,
        args,
    )
}

/// The inverses of the 16 by 16 diagonal blocks of the lower factor `l` into `inverse` (`256 * k / 16` floats).
pub(super) fn plan_invert_diagonal_blocks(l: u64, k: u32, inverse: u64) -> Result<Step, TrellisError> {
    let (l, inverse) = (address("l", l)?, address("inverse", inverse)?);
    multiple("k", k, 16)?;
    let args = Args::new().device(l).arg(int("k", k)?).device(inverse);
    step("invert_diagonal_blocks", Dim3::x(k / 16), 16, args)
}

/// Each 16-column block of `l` times the inverse of its diagonal block.
pub(super) fn plan_right_multiply_blocks(l: u64, k: u32, inverse: u64) -> Result<Step, TrellisError> {
    let (l, inverse) = (address("l", l)?, address("inverse", inverse)?);
    multiple("k", k, 16)?;
    let args = Args::new().device(l).arg(int("k", k)?).device(inverse);
    step(
        "right_multiply_blocks",
        Dim3::new(blocks(k, 128), grid_rows("k / 16", k / 16)?, 1),
        128,
        args,
    )
}

/// The root mean square of each column of `w [k, n]`.
pub(super) fn plan_column_rms(w: u64, k: u32, n: u32, out: u64) -> Result<Step, TrellisError> {
    let (w, out) = (address("w", w)?, address("out", out)?);
    let args = Args::new().device(w).arg(int("k", k)?).arg(int("n", n)?).device(out);
    step("column_rms", Dim3::x(blocks(n, 32)), 256, args)
}

/// The root mean square of each row of `w [k, n]`.
pub(super) fn plan_row_rms(w: u64, k: u32, n: u32, out: u64) -> Result<Step, TrellisError> {
    let (w, out) = (address("w", w)?, address("out", out)?);
    let args = Args::new().device(w).arg(int("k", k)?).arg(int("n", n)?).device(out);
    step("row_rms", Dim3::x(k.max(1)), 256, args)
}

/// The output-side scales: the column RMS over its mean (or as it is when the mean is zero), with the signs folded
/// in when `apply`, into `divisor` (what the weight is divided by) and `stored` (what the linear keeps).
pub(super) fn plan_output_scales(
    rms: u64,
    mean: u64,
    n: u32,
    signs: u64,
    apply: bool,
    divisor: u64,
    stored: u64,
) -> Result<Step, TrellisError> {
    let (rms, mean, signs) = (address("rms", rms)?, address("mean", mean)?, address("signs", signs)?);
    let (divisor, stored) = (address("divisor", divisor)?, address("stored", stored)?);
    let args = Args::new()
        .device(rms)
        .device(mean)
        .arg(int("n", n)?)
        .device(signs)
        .arg(i32::from(apply))
        .device(divisor)
        .device(stored);
    step("output_scales", Dim3::x(blocks(n, 256)), 256, args)
}

/// The input-side scales `su` from the row RMS and the signs.
pub(super) fn plan_input_scales(rms: u64, signs: u64, k: u32, su: u64) -> Result<Step, TrellisError> {
    let (rms, signs, su) = (address("rms", rms)?, address("signs", signs)?, address("su", su)?);
    let args = Args::new().device(rms).device(signs).arg(int("k", k)?).device(su);
    step("input_scales", Dim3::x(blocks(k, 256)), 256, args)
}

/// Gathers `samples` tiles of `w [k, n]`, at the tile coordinates in `tile_k` and `tile_n`, times `scale`, in tile
/// order into `out`.
pub(super) fn plan_gather_tiles(
    w: u64,
    n: u32,
    tile_k: u64,
    tile_n: u64,
    scale: f32,
    samples: u32,
    out: u64,
) -> Result<Step, TrellisError> {
    let (w, tile_k) = (address("w", w)?, address("tile_k", tile_k)?);
    let (tile_n, out) = (address("tile_n", tile_n)?, address("out", out)?);
    let args = Args::new()
        .device(w)
        .arg(int("n", n)?)
        .device(tile_k)
        .device(tile_n)
        .arg(scale)
        .device(out);
    step("gather_tiles", Dim3::x(samples.max(1)), 256, args)
}

/// For each candidate scale, the sampled tiles (every `stride`th of them, `per` of each) times it, into `out`.
pub(super) fn plan_candidate_tiles(
    samples: u64,
    stride: u32,
    per: u32,
    scales: u64,
    values: u64,
    out: u64,
) -> Result<Step, TrellisError> {
    let (samples, scales, out) = (
        address("samples", samples)?,
        address("scales", scales)?,
        address("out", out)?,
    );
    let args = Args::new()
        .device(samples)
        .arg(int("stride", stride)?)
        .arg(int("per", per)?)
        .device(scales)
        .arg(long("values", values)?)
        .device(out);
    step("candidate_tiles", Dim3::x(elementwise(values)), 256, args)
}

/// The mean squared error of each candidate's rounded tiles `q` against the samples, into `out`.
pub(super) fn plan_candidate_mse(
    q: u64,
    samples: u64,
    stride: u32,
    per: u32,
    scales: u64,
    candidates: u32,
    out: u64,
) -> Result<Step, TrellisError> {
    let (q, samples) = (address("q", q)?, address("samples", samples)?);
    let (scales, out) = (address("scales", scales)?, address("out", out)?);
    let args = Args::new()
        .device(q)
        .device(samples)
        .arg(int("stride", stride)?)
        .arg(int("per", per)?)
        .device(scales)
        .device(out);
    step("candidate_mse", Dim3::x(candidates.max(1)), 256, args)
}

/// The 16 rows of the strip at `first_row` of `w [k, n]`, plus the same rows of `feedback` when it is not null (the
/// uncalibrated path has none), as `n / 16` tiles at `tiles`.
pub(super) fn plan_strip_gather_sum(
    w: u64,
    feedback: u64,
    n: u32,
    first_row: u32,
    tiles: u64,
) -> Result<Step, TrellisError> {
    let (w, tiles) = (address("w", w)?, address("tiles", tiles)?);
    multiple("n", n, 16)?;
    let args = Args::new()
        .device(w)
        .device(feedback)
        .arg(int("n", n)?)
        .arg(int("first_row", first_row)?)
        .device(tiles);
    step("strip_gather_sum", Dim3::x(n / 16), 256, args)
}

/// The error of row `first_row` (`w - q`) into row `e_row` of the error block `e`.
pub(super) fn plan_strip_error(
    w: u64,
    q: u64,
    n: u32,
    first_row: u32,
    e: u64,
    e_row: u32,
) -> Result<Step, TrellisError> {
    let (w, q, e) = (address("w", w)?, address("q", q)?, address("e", e)?);
    let args = Args::new()
        .device(w)
        .device(q)
        .arg(int("n", n)?)
        .arg(int("first_row", first_row)?)
        .device(e)
        .arg(int("e_row", e_row)?);
    // one thread per element of the 16 rows
    step("strip_error", Dim3::x(blocks(n.saturating_mul(16), 256)), 256, args)
}

/// `y += alpha * x` over `count` floats.
pub(super) fn plan_axpy_floats(y: u64, x: u64, count: u64, alpha: f32) -> Result<Step, TrellisError> {
    let (y, x) = (address("y", y)?, address("x", x)?);
    let args = Args::new().device(y).device(x).arg(long("count", count)?).arg(alpha);
    step("axpy_floats", Dim3::x(elementwise(count)), 256, args)
}

/// `out = a - b` over `count` floats.
pub(super) fn plan_subtract_floats(a: u64, b: u64, out: u64, count: u64) -> Result<Step, TrellisError> {
    let (a, b, out) = (address("a", a)?, address("b", b)?, address("out", out)?);
    let args = Args::new().device(a).device(b).device(out).arg(long("count", count)?);
    step("subtract_floats", Dim3::x(elementwise(count)), 256, args)
}

/// For each column of `[k, n]` matrices, `sum(a * b) + extra * sum(a * a)`, into `out [n]`.
pub(super) fn plan_column_dots(a: u64, b: u64, extra: f32, k: u32, n: u32, out: u64) -> Result<Step, TrellisError> {
    let (a, b, out) = (address("a", a)?, address("b", b)?, address("out", out)?);
    let args = Args::new()
        .device(a)
        .device(b)
        .arg(extra)
        .arg(int("k", k)?)
        .arg(int("n", n)?)
        .device(out);
    step("column_dots", Dim3::x(blocks(n, 32)), 256, args)
}

/// For each row of `[k, n]` matrices, `sum(a * b)`, into `out [k]`.
pub(super) fn plan_row_dots(a: u64, b: u64, k: u32, n: u32, out: u64) -> Result<Step, TrellisError> {
    let (a, b, out) = (address("a", a)?, address("b", b)?, address("out", out)?);
    let args = Args::new()
        .device(a)
        .device(b)
        .arg(int("k", k)?)
        .arg(int("n", n)?)
        .device(out);
    step("row_dots", Dim3::x(k.max(1)), 256, args)
}

/// `num[c] /= dn[c]` for `n` columns.
pub(super) fn plan_column_ratio(num: u64, dn: u64, n: u32) -> Result<Step, TrellisError> {
    let (num, dn) = (address("num", num)?, address("dn", dn)?);
    let args = Args::new().device(num).device(dn).arg(int("n", n)?);
    step("column_ratio", Dim3::x(blocks(n, 256)), 256, args)
}

/// `a[r, c] *= h[r, c]`, with `damping` added to the diagonal of `h`, for the `k` by `k` matrices.
pub(super) fn plan_hessian_product(a: u64, h: u64, k: u32, damping: f32) -> Result<Step, TrellisError> {
    let (a, h) = (address("a", a)?, address("h", h)?);
    let args = Args::new().device(a).device(h).arg(int("k", k)?).arg(damping);
    step(
        "hessian_product",
        Dim3::new(blocks(k, 256), grid_rows("k", k.max(1))?, 1),
        256,
        args,
    )
}

/// Rounds `count` floats to half precision.
pub(super) fn plan_float_to_half(x: u64, count: u64, out: u64) -> Result<Step, TrellisError> {
    let (x, out) = (address("x", x)?, address("out", out)?);
    let args = Args::new().device(x).arg(long("count", count)?).device(out);
    step("float_to_half", Dim3::x(elementwise(count)), 256, args)
}

/// `x *= s` (or `/= s`) over `count` floats.
pub(super) fn plan_scale_scalar(x: u64, count: u64, s: f32, scaling: Scaling) -> Result<Step, TrellisError> {
    let x = address("x", x)?;
    let args = Args::new()
        .device(x)
        .arg(long("count", count)?)
        .arg(s)
        .arg(scaling as i32);
    step("scale_scalar", Dim3::x(elementwise(count)), 256, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(step: &Step) -> ((u32, u32), u32) {
        let launch = step.launch();
        ((launch.grid.x, launch.grid.y), launch.block.x)
    }

    #[test]
    fn grids_follow_the_shape_of_the_kernels_work() {
        let weight = plan_weight_to_float(64, 512, 1024, 128).unwrap();
        assert_eq!(geometry(&weight), ((32, 16), 256));
        let strided = plan_strided_mean(64, 1024, 1025, 128).unwrap();
        assert_eq!(geometry(&strided), ((1, 1), 1024));
        let triangle = plan_triangle(64, 1024, Triangle::ClearUpper).unwrap();
        assert_eq!(geometry(&triangle), ((32, 128), 256));
        let inverse = plan_invert_diagonal_blocks(64, 1024, 128).unwrap();
        assert_eq!(geometry(&inverse), ((64, 1), 16));
        let multiply = plan_right_multiply_blocks(64, 1024, 128).unwrap();
        assert_eq!(geometry(&multiply), ((8, 64), 128));
        let scale = plan_scale_floats(64, 1024, 512, 128, Axis::Columns, Scaling::Divide).unwrap();
        assert_eq!(geometry(&scale), ((2, 1024), 256));
        let rows = plan_row_rms(64, 1024, 512, 128).unwrap();
        assert_eq!(geometry(&rows), ((1024, 1), 256));
        let columns = plan_column_rms(64, 1024, 512, 128).unwrap();
        assert_eq!(geometry(&columns), ((16, 1), 256));
        let product = plan_hessian_product(64, 128, 1024, 0.5).unwrap();
        assert_eq!(geometry(&product), ((4, 1024), 256));
        let gather = plan_strip_gather_sum(64, 0, 512, 16, 128).unwrap();
        assert_eq!(geometry(&gather), ((32, 1), 256));
        let error = plan_strip_error(64, 128, 512, 16, 192, 3).unwrap();
        assert_eq!(geometry(&error), ((32, 1), 256));
    }

    #[test]
    fn a_grid_taller_than_the_device_allows_is_refused() {
        assert!(matches!(
            plan_hessian_product(64, 128, 70_000, 0.5),
            Err(TrellisError::GridTooTall { .. })
        ));
        assert!(matches!(
            plan_scale_floats(64, 70_000, 4, 128, Axis::Rows, Scaling::Multiply),
            Err(TrellisError::GridTooTall { .. })
        ));
    }

    #[test]
    fn a_null_address_is_refused_except_the_feedback_of_a_strip() {
        assert!(matches!(
            plan_copy_floats(0, 128, 4),
            Err(TrellisError::NullAddress("src"))
        ));
        assert!(plan_strip_gather_sum(64, 0, 512, 0, 128).is_ok());
        assert!(matches!(
            plan_strip_gather_sum(0, 0, 512, 0, 128),
            Err(TrellisError::NullAddress("w"))
        ));
    }

    #[test]
    fn the_diagonal_blocks_need_a_multiple_of_sixteen() {
        assert!(matches!(
            plan_invert_diagonal_blocks(64, 1020, 128),
            Err(TrellisError::NotMultiple { .. })
        ));
    }
}
