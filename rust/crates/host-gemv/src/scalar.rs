//! The portable scalar variant: the exact reference twin of the numeric contract. It follows the contract
//! statement by statement and is deliberately not optimised.

use std::ops::Range;

use crate::contract::{ACCUMULATORS, BLOCK, E2M1_X2, GROUP, coefficient, decode_e4m3, reduce8, x_position};
use crate::{Nvfp4View, QuantizedRows};

/// Output rows `rows_out` for every activation row; `y[m * rows_out.len() + (n - rows_out.start)]`.
pub(crate) fn gemv(w: &Nvfp4View<'_>, x: &QuantizedRows, rows_out: Range<usize>, y: &mut [f32]) {
    let len = rows_out.len();
    for m in 0..x.rows() {
        for (i, n) in rows_out.clone().enumerate() {
            y[m * len + i] = row_dot(w, x, m, n);
        }
    }
}

fn row_dot(w: &Nvfp4View<'_>, x: &QuantizedRows, m: usize, n: usize) -> f32 {
    let codes = &w.codes()[n * w.code_stride()..][..w.code_stride()];
    let scales = &w.scales()[n * w.scale_stride()..][..w.scale_stride()];
    let q = &x.q()[m * x.q_stride()..][..x.q_stride()];
    let xs = &x.scales()[m * x.scale_stride()..][..x.scale_stride()];
    let mut a = [0.0f32; ACCUMULATORS];
    for g in 0..w.k() / GROUP {
        let mut dot = 0i32;
        for i in 0..GROUP {
            let k = g * GROUP + i;
            let byte = codes[k / 2];
            let code = if k.is_multiple_of(2) { byte & 15 } else { byte >> 4 };
            dot += i32::from(E2M1_X2[usize::from(code)]) * i32::from(q[x_position(k)]);
        }
        let t = decode_e4m3(scales[g]) * w.global();
        let c = coefficient(t, xs[g * GROUP / BLOCK]);
        // |dot| <= 16 * 12 * 127, exact in f32.
        a[g % ACCUMULATORS] = (dot as f32).mul_add(c, a[g % ACCUMULATORS]);
    }
    reduce8(a)
}
