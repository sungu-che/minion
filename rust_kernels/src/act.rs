//! Gated activations in one pass, no temporaries.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::par::{self, Cptr, Ptr};
use crate::simd;

/// Language MLP: `u` is `[n, 2·inter]` = `[gate | up]`;
/// `out[i, j] = u[i, inter + j] · silu(u[i, j])`.
pub unsafe fn swiglu_split(u: *const f32, n: usize, inter: usize, out: *mut f32) {
    let (u, o) = (Cptr(u), Ptr(out));
    par::chunks(n, 1, |a, b| {
        for i in a..b {
            let gate = u.get().add(i * 2 * inter);
            let up = gate.add(inter);
            silu_mul_row(gate, up, inter, o.get().add(i * inter));
        }
    });
}

/// Vision MLP: `out = silu(gate) · up`, element-wise over `len` values.
pub unsafe fn silu_mul(gate: *const f32, up: *const f32, len: usize, out: *mut f32) {
    let (g, u, o) = (Cptr(gate), Cptr(up), Ptr(out));
    par::chunks(len, 1 << 14, |a, b| silu_mul_row(g.get().add(a), u.get().add(a), b - a, o.get().add(a)));
}

unsafe fn silu_mul_row(gate: *const f32, up: *const f32, len: usize, out: *mut f32) {
    let mut j = 0;
    #[cfg(target_arch = "x86_64")]
    if simd::fast() {
        j = silu_mul_avx2(gate, up, len, out);
    }
    while j < len {
        *out.add(j) = *up.add(j) * simd::silu_scalar(*gate.add(j));
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_mul_avx2(gate: *const f32, up: *const f32, len: usize, out: *mut f32) -> usize {
    let mut j = 0;
    while j + 8 <= len {
        let s = simd::silu256(_mm256_loadu_ps(gate.add(j)));
        _mm256_storeu_ps(out.add(j), _mm256_mul_ps(_mm256_loadu_ps(up.add(j)), s));
        j += 8;
    }
    j
}
