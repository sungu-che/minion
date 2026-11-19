//! Row normalisations, one pass over the data per row (statistics in f32).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::par::{self, Cptr, Ptr};
use crate::simd;

/// Σ x² of one row, eight lanes then a fixed reduction.
unsafe fn sum_sq(x: *const f32, d: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd::fast() {
        return sum_sq_avx2(x, d);
    }
    simd::dot_scalar(x, x, d)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_avx2(x: *const f32, d: usize) -> f32 {
    let mut a = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= d {
        let v = _mm256_loadu_ps(x.add(j));
        a = _mm256_fmadd_ps(v, v, a);
        j += 8;
    }
    let mut s = simd::hsum(a);
    while j < d {
        s += *x.add(j) * *x.add(j);
        j += 1;
    }
    s
}

unsafe fn sum(x: *const f32, d: usize) -> f32 {
    let mut lanes = [0f32; 8];
    for j in 0..d {
        lanes[j % 8] += *x.add(j);
    }
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

/// `out = x · (1 / sqrt(mean(x²) + eps)) · w`; `out` may be `x`.
pub unsafe fn rms_norm(x: *const f32, n: usize, d: usize, w: *const f32, eps: f32, out: *mut f32) {
    let (x, w, o) = (Cptr(x), Cptr(w), Ptr(out));
    par::chunks(n, (1 << 14) / d.max(1) + 1, |a, b| {
        for i in a..b {
            let xr = x.get().add(i * d);
            let or = o.get().add(i * d);
            let scale = 1.0 / (sum_sq(xr, d) / d as f32 + eps).sqrt();
            for j in 0..d {
                *or.add(j) = *xr.add(j) * scale * *w.get().add(j);
            }
        }
    });
}

/// `x += rms_norm(y) · w` (the post-norm residual of the language layers).
pub unsafe fn rms_norm_add(x: *mut f32, y: *const f32, n: usize, d: usize, w: *const f32, eps: f32) {
    let (x, y, w) = (Ptr(x), Cptr(y), Cptr(w));
    par::chunks(n, (1 << 14) / d.max(1) + 1, |a, b| {
        for i in a..b {
            let yr = y.get().add(i * d);
            let xr = x.get().add(i * d);
            let scale = 1.0 / (sum_sq(yr, d) / d as f32 + eps).sqrt();
            for j in 0..d {
                *xr.add(j) += *yr.add(j) * scale * *w.get().add(j);
            }
        }
    });
}

/// `out = (x − mean) / sqrt(var + eps) · w + b`; `out` may be `x`.
pub unsafe fn layer_norm(x: *const f32, n: usize, d: usize, w: *const f32, b: *const f32, eps: f32, out: *mut f32) {
    let (x, w, b, o) = (Cptr(x), Cptr(w), Cptr(b), Ptr(out));
    par::chunks(n, (1 << 14) / d.max(1) + 1, |a, e| {
        let mut xc = vec![0f32; d];
        for i in a..e {
            let xr = x.get().add(i * d);
            let or = o.get().add(i * d);
            let mu = sum(xr, d) / d as f32;
            for j in 0..d {
                xc[j] = *xr.add(j) - mu;
            }
            let var = sum_sq(xc.as_ptr(), d) / d as f32;
            let den = (var + eps).sqrt();
            for j in 0..d {
                *or.add(j) = xc[j] / den * *w.get().add(j) + *b.get().add(j);
            }
        }
    });
}
