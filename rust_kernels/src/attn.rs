//! Attention with the rotary embedding of each tower fused in.
//!
//! * language: rotation of adjacent pairs `(x[2i], x[2i+1])`, grouped-query
//!   heads (query heads `2g`, `2g+1` read key/value head `g`), the key/value
//!   rows written straight into the caller's cache, causal or block masking;
//! * vision: per-head RMSNorm of q and k, rotation of the two halves
//!   `(x[i], x[i+32])`, full attention.
//!
//! The core is a streaming ("flash") attention over panels of 64 keys with an
//! online softmax: the running output is rescaled when the row maximum grows,
//! so no `N × N` score matrix is ever stored.  Keys are read transposed
//! (`kt[dim][key]`), which lets eight keys share one vector of scores: the
//! language cache keeps its keys transposed for that reason, the vision tower
//! transposes them once per call.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::par::{self, Buf, Cptr, Ptr};
use crate::simd;

const KP: usize = 64; // keys per panel
const QB_LM: usize = 16;
const QB_VIS: usize = 32;

pub const ERR_CAPACITY: i32 = -5;

struct Kv {
    kt: *const f32, // key j, dim e at kt[e * kt_stride + j]
    kt_stride: usize,
    v: *const f32, // key j, dim e at v[j * v_stride + e]
    v_stride: usize,
}

/// Streaming attention of `q.len()` query rows of one head.
unsafe fn attend(q: &[*const f32], limit: &[usize], out: &[*mut f32], kv: &Kv, d: usize, scale: f32, scratch: *mut f32) {
    let rows = q.len();
    let s = scratch; // [KP]
    let acc = scratch.add(KP); // [rows][d]
    let mut m = [f32::NEG_INFINITY; QB_VIS];
    let mut l = [0f32; QB_VIS];
    for r in 0..rows {
        std::ptr::write_bytes(acc.add(r * d), 0, d);
    }
    let kmax = limit.iter().copied().max().unwrap_or(0);
    let mut j0 = 0;
    while j0 < kmax {
        for r in 0..rows {
            if limit[r] <= j0 {
                continue;
            }
            let valid = (limit[r] - j0).min(KP);
            scores(q[r], kv.kt.add(j0), kv.kt_stride, d, scale, s);
            let mut mx = m[r];
            for j in 0..valid {
                mx = mx.max(*s.add(j));
            }
            let corr = if m[r] == f32::NEG_INFINITY { 0.0 } else { simd::exp_scalar(m[r] - mx) };
            let sum = exp_shift(s, valid, mx);
            l[r] = l[r] * corr + sum;
            m[r] = mx;
            accumulate(acc.add(r * d), corr, s, kv.v.add(j0 * kv.v_stride), kv.v_stride, valid, d);
        }
        j0 += KP;
    }
    for r in 0..rows {
        let inv = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
        for e in 0..d {
            *out[r].add(e) = *acc.add(r * d + e) * inv;
        }
    }
}

/// `s[j] = scale · Σ_e q[e] · kt[e][j]` for the 64 keys of a panel.
unsafe fn scores(q: *const f32, kt: *const f32, stride: usize, d: usize, scale: f32, s: *mut f32) {
    #[cfg(target_arch = "x86_64")]
    if simd::fast() {
        scores_avx2(q, kt, stride, d, scale, s);
        return;
    }
    for j in 0..KP {
        let mut a = 0f32;
        for e in 0..d {
            a += *q.add(e) * *kt.add(e * stride + j);
        }
        *s.add(j) = a * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores_avx2(q: *const f32, kt: *const f32, stride: usize, d: usize, scale: f32, s: *mut f32) {
    let mut a = [_mm256_setzero_ps(); 8];
    for e in 0..d {
        let qv = _mm256_set1_ps(*q.add(e));
        let row = kt.add(e * stride);
        for (c, ac) in a.iter_mut().enumerate() {
            *ac = _mm256_fmadd_ps(qv, _mm256_loadu_ps(row.add(c * 8)), *ac);
        }
    }
    let sc = _mm256_set1_ps(scale);
    for (c, ac) in a.iter().enumerate() {
        _mm256_storeu_ps(s.add(c * 8), _mm256_mul_ps(*ac, sc));
    }
}

/// `s[j] = exp(s[j] − mx)` for `j < n`, returns their sum.
unsafe fn exp_shift(s: *mut f32, n: usize, mx: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd::fast() {
        return exp_shift_avx2(s, n, mx);
    }
    let mut sum = 0f32;
    for j in 0..n {
        let e = simd::exp_scalar(*s.add(j) - mx);
        *s.add(j) = e;
        sum += e;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_shift_avx2(s: *mut f32, n: usize, mx: f32) -> f32 {
    let m = _mm256_set1_ps(mx);
    let mut acc = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= n {
        let e = simd::exp256(_mm256_sub_ps(_mm256_loadu_ps(s.add(j)), m));
        _mm256_storeu_ps(s.add(j), e);
        acc = _mm256_add_ps(acc, e);
        j += 8;
    }
    let mut sum = simd::hsum(acc);
    while j < n {
        let e = simd::exp_scalar(*s.add(j) - mx);
        *s.add(j) = e;
        sum += e;
        j += 1;
    }
    sum
}

/// `o = o · corr + Σ_{j<n} p[j] · v[j]`
unsafe fn accumulate(o: *mut f32, corr: f32, p: *const f32, v: *const f32, stride: usize, n: usize, d: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd::fast() && d % 64 == 0 {
        accumulate_avx2(o, corr, p, v, stride, n, d);
        return;
    }
    for e in 0..d {
        let mut a = *o.add(e) * corr;
        for j in 0..n {
            a += *p.add(j) * *v.add(j * stride + e);
        }
        *o.add(e) = a;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_avx2(o: *mut f32, corr: f32, p: *const f32, v: *const f32, stride: usize, n: usize, d: usize) {
    let c = _mm256_set1_ps(corr);
    let mut h = 0;
    while h < d {
        let mut a = [_mm256_setzero_ps(); 8];
        for (i, ai) in a.iter_mut().enumerate() {
            *ai = _mm256_mul_ps(_mm256_loadu_ps(o.add(h + i * 8)), c);
        }
        for j in 0..n {
            let pj = _mm256_set1_ps(*p.add(j));
            let vr = v.add(j * stride + h);
            for (i, ai) in a.iter_mut().enumerate() {
                *ai = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vr.add(i * 8)), *ai);
            }
        }
        for (i, ai) in a.iter().enumerate() {
            _mm256_storeu_ps(o.add(h + i * 8), *ai);
        }
        h += 64;
    }
}

// ------------------------------------------------------------------------
// language
// ------------------------------------------------------------------------
pub const LM_HEADS: usize = 16;
pub const LM_KV: usize = 8;
pub const LM_DIM: usize = 128;

/// Rotate adjacent pairs of `heads` rows of 128 in place (`cos`/`sin` are `[64]`).
unsafe fn rope_pairs(x: *mut f32, heads: usize, cos: *const f32, sin: *const f32) {
    for h in 0..heads {
        let r = x.add(h * LM_DIM);
        for i in 0..LM_DIM / 2 {
            let a = *r.add(2 * i);
            let b = *r.add(2 * i + 1);
            let c = *cos.add(i);
            let s = *sin.add(i);
            *r.add(2 * i) = a * c - b * s;
            *r.add(2 * i + 1) = b * c + a * s;
        }
    }
}

/// q `[t, 16·128]` and k `[t, 8·128]` are rotated in place; k / v go to the
/// caches at keys `past..past+t` (`kt` `[8, 128, cap]` transposed,
/// `vc` `[8, cap, 128]`); the output is `[t, 2048]`.  `cap` is a multiple of 64.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lm_attention(
    q: *mut f32,
    k: *mut f32,
    v: *const f32,
    cos: *const f32,
    sin: *const f32,
    kt: *mut f32,
    vc: *mut f32,
    cap: usize,
    past: usize,
    t: usize,
    causal: bool,
    out: *mut f32,
) -> i32 {
    if past + t > cap || cap % KP != 0 {
        return ERR_CAPACITY;
    }
    let (qp, kp, vp) = (Ptr(q), Ptr(k), Cptr(v));
    let (cs, sn) = (Cptr(cos), Cptr(sin));
    let (ktp, vcp) = (Ptr(kt), Ptr(vc));
    par::chunks(t, 4, |a, b| {
        for i in a..b {
            let c = cs.get().add(i * 64);
            let s = sn.get().add(i * 64);
            rope_pairs(qp.get().add(i * LM_HEADS * LM_DIM), LM_HEADS, c, s);
            rope_pairs(kp.get().add(i * LM_KV * LM_DIM), LM_KV, c, s);
            for g in 0..LM_KV {
                let src = kp.get().add((i * LM_KV + g) * LM_DIM);
                let col = ktp.get().add(g * LM_DIM * cap + past + i);
                for e in 0..LM_DIM {
                    *col.add(e * cap) = *src.add(e);
                }
                let row = vcp.get().add((g * cap + past + i) * LM_DIM);
                std::ptr::copy_nonoverlapping(vp.get().add((i * LM_KV + g) * LM_DIM), row, LM_DIM);
            }
        }
    });
    // the query rows of kv head g: member 0 tokens 0..t, then member 1
    let rows_per_head = 2 * t;
    let qblocks = rows_per_head.div_ceil(QB_LM);
    let scale = (LM_DIM as f32).powf(-0.5);
    let op = Ptr(out);
    par::run(LM_KV * qblocks, |task| {
        let g = task / qblocks;
        let r0 = (task % qblocks) * QB_LM;
        let r1 = (r0 + QB_LM).min(rows_per_head);
        let mut qs = [std::ptr::null::<f32>(); QB_LM];
        let mut os = [std::ptr::null_mut::<f32>(); QB_LM];
        let mut lim = [0usize; QB_LM];
        for (n, r) in (r0..r1).enumerate() {
            let head = g * 2 + r / t;
            let i = r % t;
            qs[n] = qp.get().add((i * LM_HEADS + head) * LM_DIM);
            os[n] = op.get().add((i * LM_HEADS + head) * LM_DIM);
            lim[n] = if causal { past + i + 1 } else { past + t };
        }
        let rows = r1 - r0;
        let kv = Kv {
            kt: ktp.get().add(g * LM_DIM * cap),
            kt_stride: cap,
            v: vcp.get().add(g * cap * LM_DIM),
            v_stride: LM_DIM,
        };
        par::with_scratch(KP + QB_LM * LM_DIM, |buf| {
            attend(&qs[..rows], &lim[..rows], &os[..rows], &kv, LM_DIM, scale, buf);
        });
    });
    0
}

// ------------------------------------------------------------------------
// vision
// ------------------------------------------------------------------------
pub const VIS_HEADS: usize = 16;
pub const VIS_DIM: usize = 64;
const VIS_WIDTH: usize = VIS_HEADS * VIS_DIM;

/// RMSNorm then half-split rotation of one 64-wide head row, in place.
unsafe fn norm_rope_half(x: *mut f32, w: *const f32, eps: f32, cos: *const f32, sin: *const f32) {
    let mut lanes = [0f32; 8];
    for j in 0..VIS_DIM {
        let v = *x.add(j);
        lanes[j % 8] += v * v;
    }
    let ss = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    let scale = 1.0 / (ss / VIS_DIM as f32 + eps).sqrt();
    let mut h = [0f32; VIS_DIM];
    for j in 0..VIS_DIM {
        h[j] = *x.add(j) * scale * *w.add(j);
    }
    let half = VIS_DIM / 2;
    for j in 0..VIS_DIM {
        let rot = if j < half { -h[j + half] } else { h[j - half] };
        *x.add(j) = h[j] * *cos.add(j) + rot * *sin.add(j);
    }
}

/// `qkv` `[n, 3, 16, 64]` (q and k are normalised and rotated in place),
/// cos / sin `[n, 64]` → attention output `[n, 1024]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vis_attention(
    qkv: *mut f32,
    q_norm: *const f32,
    k_norm: *const f32,
    eps: f32,
    cos: *const f32,
    sin: *const f32,
    n: usize,
    out: *mut f32,
) -> i32 {
    if n == 0 {
        return 0;
    }
    let row = 3 * VIS_WIDTH;
    let x = Ptr(qkv);
    let (qn, kn) = (Cptr(q_norm), Cptr(k_norm));
    let (cs, sn) = (Cptr(cos), Cptr(sin));
    // keys transposed per head, padded to whole panels: kt[h][e][j]
    let npad = n.div_ceil(KP) * KP;
    let kt = Buf::new(VIS_HEADS * VIS_DIM * npad);
    let ktp = Ptr(kt.ptr());
    par::chunks(n, 64, |a, b| {
        for i in a..b {
            let c = cs.get().add(i * VIS_DIM);
            let s = sn.get().add(i * VIS_DIM);
            for h in 0..VIS_HEADS {
                norm_rope_half(x.get().add(i * row + h * VIS_DIM), qn.get(), eps, c, s);
                let kr = x.get().add(i * row + VIS_WIDTH + h * VIS_DIM);
                norm_rope_half(kr, kn.get(), eps, c, s);
                let col = ktp.get().add(h * VIS_DIM * npad + i);
                for e in 0..VIS_DIM {
                    *col.add(e * npad) = *kr.add(e);
                }
            }
        }
    });
    let qblocks = n.div_ceil(QB_VIS);
    let scale = (VIS_DIM as f32).powf(-0.5);
    let op = Ptr(out);
    par::run(VIS_HEADS * qblocks, |task| {
        let h = task / qblocks;
        let i0 = (task % qblocks) * QB_VIS;
        let i1 = (i0 + QB_VIS).min(n);
        let mut qs = [std::ptr::null::<f32>(); QB_VIS];
        let mut os = [std::ptr::null_mut::<f32>(); QB_VIS];
        let lim = [n; QB_VIS];
        for (r, i) in (i0..i1).enumerate() {
            qs[r] = x.get().add(i * row + h * VIS_DIM);
            os[r] = op.get().add(i * VIS_WIDTH + h * VIS_DIM);
        }
        let rows = i1 - i0;
        let kv = Kv {
            kt: ktp.get().add(h * VIS_DIM * npad),
            kt_stride: npad,
            v: x.get().add(2 * VIS_WIDTH + h * VIS_DIM),
            v_stride: row,
        };
        par::with_scratch(KP + QB_VIS * VIS_DIM, |buf| {
            attend(&qs[..rows], &lim[..rows], &os[..rows], &kv, VIS_DIM, scale, buf);
        });
    });
    drop(kt);
    0
}
