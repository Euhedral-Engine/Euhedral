//! The launch planners: each function turns a shape and device addresses into the steps one operation runs.

use euhedral_gpu::{Args, Dim3, Launch};

use super::rate::{Rate, TILE_WEIGHTS};
use super::{Step, TrellisError};
use crate::kernels::Module;

/// Bytes of one float.
const FLOAT_BYTES: u64 = 4;
/// Bytes of one 16-bit state.
const STATE_BYTES: u64 = 2;
/// The most blocks of an elementwise launch.
const MOST_ELEMENTWISE_BLOCKS: u64 = 1 << 20;
/// The most blocks of a reduction launch: each block adds into one double, so more blocks only add contention.
const MOST_REDUCTION_BLOCKS: u64 = 1024;

pub(super) fn rate_kernel(kind: &str, rate: Rate) -> Result<&'static str, TrellisError> {
    let frac = if rate.is_half_step() { "_frac" } else { "" };
    let name = format!("{kind}{frac}_k{}", rate.lower());
    Module::Trellis
        .kernels()
        .iter()
        .copied()
        .find(|listed| *listed == name)
        .ok_or(TrellisError::NoKernel(name))
}

fn step(kernel: &str, grid: Dim3, block: u32, args: Args) -> Result<Step, TrellisError> {
    let launch = Launch {
        grid,
        block: Dim3::x(block),
        shared_bytes: 0,
        programmatic: false,
    };
    Step::new(Module::Trellis, kernel, launch, args).ok_or_else(|| TrellisError::NoKernel(kernel.to_owned()))
}

fn address(name: &'static str, value: u64) -> Result<u64, TrellisError> {
    if value == 0 {
        Err(TrellisError::NullAddress(name))
    } else {
        Ok(value)
    }
}

/// `value` as the kernel's signed 32-bit parameter.
fn int(name: &'static str, value: u32) -> Result<i32, TrellisError> {
    i32::try_from(value).map_err(|_| TrellisError::TooLarge {
        name,
        value: u64::from(value),
    })
}

/// `value` as the kernel's signed 64-bit parameter.
fn long(name: &'static str, value: u64) -> Result<i64, TrellisError> {
    i64::try_from(value).map_err(|_| TrellisError::TooLarge { name, value })
}

/// `value`, when it is a positive multiple of `unit`.
fn multiple(name: &'static str, value: u32, unit: u32) -> Result<u32, TrellisError> {
    if value == 0 || !value.is_multiple_of(unit) {
        return Err(TrellisError::NotMultiple {
            name,
            value: u64::from(value),
            unit,
        });
    }
    Ok(value)
}

/// The most blocks a grid has along its second axis.
const MOST_GRID_ROWS: u32 = 65535;

/// `blocks`, when a grid can be that tall.
fn grid_rows(name: &'static str, blocks: u32) -> Result<u32, TrellisError> {
    if blocks > MOST_GRID_ROWS {
        return Err(TrellisError::GridTooTall {
            name,
            value: u64::from(blocks),
            most: MOST_GRID_ROWS,
        });
    }
    Ok(blocks)
}

/// Blocks of an elementwise launch over `count` values, 256 threads each.
fn elementwise(count: u64) -> u32 {
    count.div_ceil(256).clamp(1, MOST_ELEMENTWISE_BLOCKS) as u32
}

/// The Viterbi encode of `count` float32 tiles `[count, 256]` at `tiles` into 16-bit states at `indices` and the
/// float32 codebook values they decode to at `decoded`. `scratch` holds `rate.scratch_bytes(scratch_tiles)` bytes.
/// The tiles run in chunks of at most `scratch_tiles`, each reusing the scratch in order on the lane; every tile is
/// independent, so the chunk size changes no result.
pub fn plan_viterbi(
    rate: Rate,
    tiles: u64,
    count: u32,
    indices: u64,
    decoded: u64,
    scratch: u64,
    scratch_tiles: u32,
) -> Result<Vec<Step>, TrellisError> {
    let (tiles, indices) = (address("tiles", tiles)?, address("indices", indices)?);
    let (decoded, scratch) = (address("decoded", decoded)?, address("scratch", scratch)?);
    int("count", count)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if scratch_tiles == 0 {
        return Err(TrellisError::NoScratchTiles);
    }
    let kernel = rate_kernel("viterbi", rate)?;
    let mut steps = Vec::with_capacity(count.div_ceil(scratch_tiles) as usize);
    let mut start = 0u32;
    while start < count {
        let n = scratch_tiles.min(count - start);
        let offset = TILE_WEIGHTS as u64 * u64::from(start);
        let args = Args::new()
            .device(tiles + FLOAT_BYTES * offset)
            .arg(int("count", n)?)
            .device(indices + STATE_BYTES * offset)
            .device(decoded + FLOAT_BYTES * offset)
            .device(scratch);
        steps.push(
            step(kernel, Dim3::x(n), rate.viterbi_threads(), args)?.with_shared_bytes(rate.viterbi_shared_bytes()),
        );
        start += n;
    }
    Ok(steps)
}

/// Packs the 16-bit states `[count, 256]` at `indices` into `[count, 16 * bits]` 16-bit words at `words`, most
/// significant bit first.
pub fn plan_pack(rate: Rate, indices: u64, count: u32, words: u64) -> Result<Vec<Step>, TrellisError> {
    let (indices, words) = (address("indices", indices)?, address("words", words)?);
    let count_arg = int("count", count)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let blocks = if rate.is_half_step() {
        count.div_ceil(128)
    } else {
        count
    };
    let args = Args::new().device(indices).arg(count_arg).device(words);
    Ok(vec![step(rate_kernel("pack", rate)?, Dim3::x(blocks), 128, args)?])
}

/// Decodes the packed words at `words` to float32 codebook values `[count, 256]` at `decoded`.
pub fn plan_decode(rate: Rate, words: u64, count: u32, decoded: u64) -> Result<Vec<Step>, TrellisError> {
    let (words, decoded) = (address("words", words)?, address("decoded", decoded)?);
    let count_arg = int("count", count)?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let args = Args::new().device(words).arg(count_arg).device(decoded);
    Ok(vec![step(rate_kernel("decode", rate)?, Dim3::x(count), 256, args)?])
}

/// In place, every 128-row block of every column of the float32 row-major `x` `[rows, cols]` becomes
/// `H128 * block / sqrt(128)` (Sylvester order). `rows` is a multiple of 128.
pub fn plan_hadamard_rows(x: u64, rows: u32, cols: u32) -> Result<Vec<Step>, TrellisError> {
    let x = address("x", x)?;
    let (rows_arg, cols_arg) = (int("rows", rows)?, int("cols", cols)?);
    if !rows.is_multiple_of(128) {
        return Err(TrellisError::NotMultiple {
            name: "rows",
            value: u64::from(rows),
            unit: 128,
        });
    }
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let args = Args::new().device(x).arg(rows_arg).arg(cols_arg);
    Ok(vec![step(
        "hadamard128_rows",
        Dim3::new(cols.div_ceil(32), rows / 128, 1),
        256,
        args,
    )?])
}

/// In place, every 128-column block of every row of the float32 row-major `x` `[rows, cols]` becomes
/// `H128 * block / sqrt(128)`. `cols` is a multiple of 128.
pub fn plan_hadamard_columns(x: u64, rows: u32, cols: u32) -> Result<Vec<Step>, TrellisError> {
    let x = address("x", x)?;
    let (rows_arg, cols_arg) = (int("rows", rows)?, int("cols", cols)?);
    if !cols.is_multiple_of(128) {
        return Err(TrellisError::NotMultiple {
            name: "cols",
            value: u64::from(cols),
            unit: 128,
        });
    }
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let grid_y = grid_rows("rows", rows)?;
    let args = Args::new().device(x).arg(rows_arg).arg(cols_arg);
    Ok(vec![step(
        "hadamard128_columns",
        Dim3::new(cols / 128, grid_y, 1),
        128,
        args,
    )?])
}

/// Gathers the 16 rows from `first_row` of the float32 `w` `[k, n]` into `n / 16` tiles `[n / 16, 256]` at `tiles`,
/// in tensor-core order.
pub fn plan_to_tiles(w: u64, k: u32, n: u32, first_row: u32, tiles: u64) -> Result<Vec<Step>, TrellisError> {
    strip(w, k, n, first_row, tiles, false)
}

/// The inverse of [`plan_to_tiles`]: scatters the tiles back into the 16 rows of `w` from `first_row`.
pub fn plan_from_tiles(tiles: u64, k: u32, n: u32, first_row: u32, w: u64) -> Result<Vec<Step>, TrellisError> {
    strip(w, k, n, first_row, tiles, true)
}

fn strip(w: u64, k: u32, n: u32, first_row: u32, tiles: u64, inverse: bool) -> Result<Vec<Step>, TrellisError> {
    let (w, tiles) = (address("w", w)?, address("tiles", tiles)?);
    multiple("n", n, 16)?;
    if first_row.checked_add(16).is_none_or(|end| end > k) {
        return Err(TrellisError::StripOutside {
            first: first_row,
            rows: k,
        });
    }
    let args = Args::new()
        .device(w)
        .arg(int("n", n)?)
        .arg(int("first_row", first_row)?)
        .device(tiles)
        .arg(i32::from(inverse));
    Ok(vec![step("strip_tiles", Dim3::x(n / 16), 256, args)?])
}

/// The operands of a reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct Reconstruction {
    /// The packed trellis `[k / 16, n / 16, 16 * bits]` words.
    pub words: u64,
    /// The fp16 row sign-and-scale vector, `[k]`.
    pub suh: u64,
    /// The fp16 column sign-and-scale vector, `[n]`.
    pub svh: u64,
    /// Input features.
    pub k: u32,
    /// Output features.
    pub n: u32,
    /// The BF16 `nn.Linear` weight `[n, k]`.
    pub out: u64,
    /// `4 * k * n` bytes.
    pub scratch: u64,
}

/// Rebuilds the BF16 `nn.Linear` weight of an exllamav3 linear:
/// `W = had_r(had_l(Q) * suh[:, None]) * svh[None, :]` with `Q` the decoded `[k, n]` matrix.
pub fn plan_reconstruct(rate: Rate, operands: &Reconstruction) -> Result<Vec<Step>, TrellisError> {
    let Reconstruction {
        words,
        suh,
        svh,
        k,
        n,
        out,
        scratch,
    } = *operands;
    let (words, suh) = (address("words", words)?, address("suh", suh)?);
    let (svh, out) = (address("svh", svh)?, address("out", out)?);
    let scratch = address("scratch", scratch)?;
    multiple("k", k, 128)?;
    multiple("n", n, 128)?;
    let (k_arg, n_arg) = (int("k", k)?, int("n", n)?);
    let tiles = u64::from(k / 16) * u64::from(n / 16);
    let tiles = i32::try_from(tiles)
        .map_err(|_| TrellisError::TooLarge {
            name: "tiles",
            value: tiles,
        })?
        .cast_unsigned();
    let scale_grid = Dim3::new(n.div_ceil(256), grid_rows("k", k)?, 1);
    let mut steps = vec![step(
        rate_kernel("reconstruct", rate)?,
        Dim3::x(tiles),
        256,
        Args::new().device(words).arg(n_arg).device(scratch),
    )?];
    steps.extend(plan_hadamard_rows(scratch, k, n)?);
    steps.push(step(
        "scale_rows_fp16",
        scale_grid,
        256,
        Args::new().device(scratch).device(suh).arg(k_arg).arg(n_arg),
    )?);
    steps.extend(plan_hadamard_columns(scratch, k, n)?);
    steps.push(step(
        "scale_columns_fp16",
        scale_grid,
        256,
        Args::new().device(scratch).device(svh).arg(k_arg).arg(n_arg),
    )?);
    steps.push(step(
        "transpose_to_bf16",
        Dim3::new(n.div_ceil(32), k.div_ceil(32), 1),
        256,
        Args::new().device(scratch).arg(k_arg).arg(n_arg).device(out),
    )?);
    Ok(steps)
}

/// The mean of squares of each 16x16 tile of the float32 row-major `w` `[k, n]`, into `out` `[k / 16, n / 16]`
/// floats in tile row-major order.
pub fn plan_tile_mean_square(w: u64, k: u32, n: u32, out: u64) -> Result<Vec<Step>, TrellisError> {
    let (w, out) = (address("w", w)?, address("out", out)?);
    multiple("k", k, 16)?;
    multiple("n", n, 16)?;
    let grid_y = grid_rows("k / 16", k / 16)?;
    let args = Args::new().device(w).arg(int("n", n)?).device(out);
    Ok(vec![step("tile_mean_square", Dim3::new(n / 16, grid_y, 1), 256, args)?])
}

/// Adds `sum(a[i] * b[i])` for `i < count`, in double, to the double at `out`. The caller zeroes `out` first
/// ([`plan_fill_floats`]). Blocks add with an atomic, so the order of the additions is not fixed.
pub fn plan_sum_products(a: u64, b: u64, count: u64, out: u64) -> Result<Vec<Step>, TrellisError> {
    let (a, b, out) = (address("a", a)?, address("b", b)?, address("out", out)?);
    let signed = long("count", count)?;
    let blocks = elementwise(count).min(MOST_REDUCTION_BLOCKS as u32);
    let args = Args::new().device(a).device(b).arg(signed).device(out);
    Ok(vec![step("sum_products", Dim3::x(blocks), 256, args)?])
}

/// Sets `count` floats at `x` to `value`.
pub fn plan_fill_floats(x: u64, count: u64, value: f32) -> Result<Vec<Step>, TrellisError> {
    let x = address("x", x)?;
    let signed = long("count", count)?;
    Ok(vec![step(
        "fill_floats",
        Dim3::x(elementwise(count)),
        256,
        Args::new().device(x).arg(signed).arg(value),
    )?])
}

/// Cleans the solution `r` `[k]` of the input-side system: a value is kept where it is finite and positive, and
/// becomes 1 elsewhere (every value when the factorization's status at `info` is not zero, which also adds one to
/// the count at `failures`).
pub fn plan_clean_row_scales(r: u64, info: u64, k: u32, failures: u64) -> Result<Vec<Step>, TrellisError> {
    let (r, info, failures) = (address("r", r)?, address("info", info)?, address("failures", failures)?);
    let k_arg = int("k", k)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let args = Args::new().device(r).device(info).arg(k_arg).device(failures);
    Ok(vec![step("clean_row_scales", Dim3::x(k.div_ceil(256)), 256, args)?])
}
