//! `y = x · Wᵀ (+ bias)` straight on the packed weights.
//!
//! The output is cut into tiles (a block of `x` rows × a range of `W` rows);
//! a tile walks the inner dimension in chunks of `KC`.  For every chunk it
//! decodes a panel of `P` weight rows into a small per-thread buffer (it stays
//! in L1/L2) and multiplies it with the `x` rows of the tile.  The weights are
//! never expanded as a whole.
//!
//! Every output value has one 8-lane accumulator per chunk and the chunks are
//! added in a fixed order, so a value does not depend on the batch size, on the
//! tile it fell into or on the number of threads.

use crate::codec::{self, QTensor};
use crate::par::{self, Cptr, Ptr};

pub const KC: usize = 512;
const P: usize = 16;
const MC: usize = 48;

pub const FLAG_ACCUMULATE: u32 = 1;

struct Job {
    t: QTensor,
    x: Cptr<f32>,
    y: Ptr<f32>,
    bias: Cptr<f32>,
    rows: usize,
    cols: usize,
    accumulate: bool,
}

pub unsafe fn qmatmul(t: &QTensor, x: *const f32, n: usize, y: *mut f32, bias: *const f32, flags: u32) -> i32 {
    if let Err(e) = t.check() {
        return e;
    }
    if !t.rows_are_supers() {
        return codec::ERR_SHAPE;
    }
    if n == 0 {
        return 0;
    }
    let rows = t.rows as usize;
    let cols = t.cols as usize;
    let job = Job {
        t: *t,
        x: Cptr(x),
        y: Ptr(y),
        bias: Cptr(bias),
        rows,
        cols,
        accumulate: flags & FLAG_ACCUMULATE != 0,
    };
    let threads = par::threads();
    let xblocks = n.div_ceil(MC);
    // enough tiles for every thread; split the weight rows only when the
    // x blocks alone are too few
    let mut wsplit = 1;
    if xblocks < 2 * threads {
        wsplit = (4 * threads).div_ceil(xblocks).min(rows.div_ceil(P)).max(1);
    }
    let nc = rows.div_ceil(wsplit).div_ceil(P) * P;
    let wranges = rows.div_ceil(nc);
    par::run(xblocks * wranges, |task| {
        let xb = task / wranges;
        let wr = task % wranges;
        let i0 = xb * MC;
        let i1 = (i0 + MC).min(n);
        let r0 = wr * nc;
        let r1 = (r0 + nc).min(rows);
        tile(&job, i0, i1, r0, r1);
    });
    0
}

unsafe fn tile(job: &Job, i0: usize, i1: usize, r0: usize, r1: usize) {
    let rows = job.rows;
    let cols = job.cols;
    let y = job.y.get();
    let bias = job.bias.get();
    for i in i0..i1 {
        let yr = y.add(i * rows);
        for r in r0..r1 {
            let b = if bias.is_null() { 0.0 } else { *bias.add(r) };
            let v = yr.add(r);
            *v = if job.accumulate { *v + b } else { b };
        }
    }
    let dense_f32 = job.t.scheme == codec::F32;
    let kc_max = KC.min(cols);
    par::with_scratch(P * kc_max, |panel| {
        let mut k0 = 0;
        while k0 < cols {
            let kc = KC.min(cols - k0);
            let mut r = r0;
            while r < r1 {
                let p = P.min(r1 - r);
                let mut wrow = [std::ptr::null::<f32>(); P];
                for j in 0..p {
                    if dense_f32 {
                        wrow[j] = (job.t.q as *const f32).add((r + j) * cols + k0);
                    } else {
                        let dst = panel.add(j * kc);
                        codec::decode_row(&job.t, r + j, k0, kc, dst);
                        wrow[j] = dst;
                    }
                }
                panel_product(job, &wrow[..p], i0, i1, r, k0, kc);
                r += p;
            }
            k0 += kc;
        }
    });
}

/// `y[i0..i1, r..r+len(w)] += x[i0..i1, k0..k0+kc] · w`
unsafe fn panel_product(job: &Job, w: &[*const f32], i0: usize, i1: usize, r: usize, k0: usize, kc: usize) {
    let rows = job.rows;
    let cols = job.cols;
    let x = job.x.get();
    let y = job.y.get();
    #[cfg(target_arch = "x86_64")]
    if crate::simd::fast() {
        panel_avx2(x, y, rows, cols, w, i0, i1, r, k0, kc);
        return;
    }
    for i in i0..i1 {
        let xi = x.add(i * cols + k0);
        for (j, &wj) in w.iter().enumerate() {
            *y.add(i * rows + r + j) += crate::simd::dot_scalar(xi, wj, kc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn panel_avx2(
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
    w: &[*const f32],
    i0: usize,
    i1: usize,
    r: usize,
    k0: usize,
    kc: usize,
) {
    use crate::simd::dots;
    let xr = |i: usize| x.add(i * cols + k0);
    let add = |i: usize, j: usize, v: f32| *y.add(i * rows + r + j) += v;
    let nw = w.len();
    let mut i = i0;
    if i1 - i0 == 1 {
        // one row (decoding): eight weight rows per pass
        let mut j = 0;
        while j + 8 <= nw {
            let o = dots::<1, 8>([xr(i)], [w[j], w[j + 1], w[j + 2], w[j + 3], w[j + 4], w[j + 5], w[j + 6], w[j + 7]], kc);
            for (b, v) in o[0].iter().enumerate() {
                add(i, j + b, *v);
            }
            j += 8;
        }
        while j < nw {
            add(i, j, dots::<1, 1>([xr(i)], [w[j]], kc)[0][0]);
            j += 1;
        }
        return;
    }
    while i + 3 <= i1 {
        let xs = [xr(i), xr(i + 1), xr(i + 2)];
        let mut j = 0;
        while j + 4 <= nw {
            let o = dots::<3, 4>(xs, [w[j], w[j + 1], w[j + 2], w[j + 3]], kc);
            for a in 0..3 {
                for b in 0..4 {
                    add(i + a, j + b, o[a][b]);
                }
            }
            j += 4;
        }
        while j < nw {
            let o = dots::<3, 1>(xs, [w[j]], kc);
            for a in 0..3 {
                add(i + a, j, o[a][0]);
            }
            j += 1;
        }
        i += 3;
    }
    while i < i1 {
        let mut j = 0;
        while j + 4 <= nw {
            let o = dots::<1, 4>([xr(i)], [w[j], w[j + 1], w[j + 2], w[j + 3]], kc);
            for b in 0..4 {
                add(i, j + b, o[0][b]);
            }
            j += 4;
        }
        while j < nw {
            add(i, j, dots::<1, 1>([xr(i)], [w[j]], kc)[0][0]);
            j += 1;
        }
        i += 1;
    }
}
