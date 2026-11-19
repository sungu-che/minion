//! AVX2 + FMA building blocks and the switch between them and the scalar code.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::atomic::{AtomicBool, Ordering};

static FORCE_SCALAR: AtomicBool = AtomicBool::new(false);

pub fn set_scalar(on: bool) {
    FORCE_SCALAR.store(on, Ordering::Relaxed);
}

/// The CPU has what the fast paths need.
pub fn cpu_fast() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The fast paths may run (present and not switched off).
#[inline]
pub fn fast() -> bool {
    cpu_fast() && !FORCE_SCALAR.load(Ordering::Relaxed)
}

/// Sum of the eight lanes, always in the same order.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn hmax(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// `M × N` dot products of length `k`: `out[a][b] = x[a] · w[b]`.
///
/// Every product has one eight-lane accumulator of its own, so its value does
/// not depend on `M`, `N`, or on what else is computed next to it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dots<const M: usize, const N: usize>(x: [*const f32; M], w: [*const f32; N], k: usize) -> [[f32; N]; M] {
    let mut acc = [[_mm256_setzero_ps(); N]; M];
    let k8 = k & !7;
    let mut j = 0;
    while j < k8 {
        let mut xv = [_mm256_setzero_ps(); M];
        for a in 0..M {
            xv[a] = _mm256_loadu_ps(x[a].add(j));
        }
        for b in 0..N {
            let wv = _mm256_loadu_ps(w[b].add(j));
            for a in 0..M {
                acc[a][b] = _mm256_fmadd_ps(xv[a], wv, acc[a][b]);
            }
        }
        j += 8;
    }
    let mut out = [[0f32; N]; M];
    for a in 0..M {
        for b in 0..N {
            let mut s = hsum(acc[a][b]);
            for t in k8..k {
                s += x[a].add(t).read_unaligned() * w[b].add(t).read_unaligned();
            }
            out[a][b] = s;
        }
    }
    out
}

/// The scalar twin of [`dots`] for one pair (eight lanes, then the tail).
#[inline]
pub unsafe fn dot_scalar(x: *const f32, w: *const f32, k: usize) -> f32 {
    let mut acc = [0f32; 8];
    let k8 = k & !7;
    let mut j = 0;
    while j < k8 {
        for l in 0..8 {
            acc[l] += x.add(j + l).read_unaligned() * w.add(j + l).read_unaligned();
        }
        j += 8;
    }
    let mut s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    for t in k8..k {
        s += x.add(t).read_unaligned() * w.add(t).read_unaligned();
    }
    s
}

pub const EXP_HI: f32 = 88.0;
pub const EXP_LO: f32 = -87.336_54; // below this the result is zero (no denormals)

/// `exp(x)` to about one ulp (Cephes single-precision polynomial).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp256(x: __m256) -> __m256 {
    let lo = _mm256_set1_ps(EXP_LO);
    let dead = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
    let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(EXP_HI)), lo);
    let fx = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_375), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
    let r2 = _mm256_mul_ps(r, r);
    let mut p = _mm256_set1_ps(1.987_569_15e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_950_7e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_451_907_3e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_589_4e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_545_9e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.000_000_120_1e-1));
    let y = _mm256_add_ps(_mm256_fmadd_ps(p, r2, r), _mm256_set1_ps(1.0));
    let n = _mm256_cvtps_epi32(fx);
    let pow = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    _mm256_andnot_ps(dead, _mm256_mul_ps(y, _mm256_castsi256_ps(pow)))
}

#[inline]
pub fn exp_scalar(x: f32) -> f32 {
    if x < EXP_LO {
        0.0
    } else {
        x.min(EXP_HI).exp()
    }
}

/// `x · sigmoid(x)`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu256(x: __m256) -> __m256 {
    let e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + exp_scalar(-x))
}
