//! Native CPU kernels of the gravity-ocr runtime.
//!
//! The library is loaded with `ctypes` by `gravity_ocr/native.py`; every
//! array argument is the address of a numpy buffer (or of a memory-mapped
//! package part), so nothing is copied on the way in or out.  All functions
//! are `extern "C"`, take sizes as `size_t` and report errors as negative
//! return codes.  `native.py` checks every kernel against the numpy code once
//! when the library is loaded and falls back to numpy when they disagree.

#![allow(clippy::missing_safety_doc)]

pub mod act;
pub mod attn;
pub mod codec;
pub mod image;
pub mod matmul;
pub mod norm;
pub mod par;
pub mod simd;

use codec::QTensor;

/// ABI of the exported functions; `gravity_ocr/native.py` refuses a library whose version differs.
pub const VERSION: u32 = 2;

#[no_mangle]
pub extern "C" fn gk_version() -> u32 {
    VERSION
}

/// bit 0: the CPU has AVX2 + FMA + F16C; bit 1: the vector paths are in use.
#[no_mangle]
pub extern "C" fn gk_features() -> u32 {
    (simd::cpu_fast() as u32) | ((simd::fast() as u32) << 1)
}

/// Size the worker pool (`0`: `GRAVITY_OCR_THREADS` or every logical CPU).
#[no_mangle]
pub extern "C" fn gk_set_threads(n: u32) -> u32 {
    par::set_threads(n as usize) as u32
}

#[no_mangle]
pub extern "C" fn gk_threads() -> u32 {
    par::threads() as u32
}

/// Force the portable scalar code (for testing the fallback).
#[no_mangle]
pub extern "C" fn gk_set_scalar(on: i32) {
    simd::set_scalar(on != 0)
}

#[no_mangle]
pub unsafe extern "C" fn gk_check(t: *const QTensor) -> i32 {
    match (*t).check() {
        Ok(()) => 0,
        Err(e) => e,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gk_dequant(t: *const QTensor, out: *mut f32) -> i32 {
    let t = &*t;
    if let Err(e) = t.check() {
        return e;
    }
    codec::dequant(t, out);
    0
}

/// Values `[i0, i0 + len)` of the flattened tensor (the same bits as that slice of `gk_dequant`).
#[no_mangle]
pub unsafe extern "C" fn gk_dequant_range(t: *const QTensor, i0: usize, len: usize, out: *mut f32) -> i32 {
    let t = &*t;
    if let Err(e) = t.check() {
        return e;
    }
    if i0 + len > t.numel() {
        return codec::ERR_SHAPE;
    }
    codec::dequant_range(t, i0, len, out);
    0
}

/// `y = x · Wᵀ + bias` (`flags & 1`: `y += …`); `x` `[n, cols]`, `y` `[n, rows]`.
#[no_mangle]
pub unsafe extern "C" fn gk_qmatmul(t: *const QTensor, x: *const f32, n: usize, y: *mut f32, bias: *const f32, flags: u32) -> i32 {
    matmul::qmatmul(&*t, x, n, y, bias, flags)
}

#[no_mangle]
pub unsafe extern "C" fn gk_rms_norm(x: *const f32, n: usize, d: usize, w: *const f32, eps: f32, out: *mut f32) {
    norm::rms_norm(x, n, d, w, eps, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_rms_norm_add(x: *mut f32, y: *const f32, n: usize, d: usize, w: *const f32, eps: f32) {
    norm::rms_norm_add(x, y, n, d, w, eps)
}

#[no_mangle]
pub unsafe extern "C" fn gk_layer_norm(x: *const f32, n: usize, d: usize, w: *const f32, b: *const f32, eps: f32, out: *mut f32) {
    norm::layer_norm(x, n, d, w, b, eps, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_swiglu_split(u: *const f32, n: usize, inter: usize, out: *mut f32) {
    act::swiglu_split(u, n, inter, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_silu_mul(gate: *const f32, up: *const f32, len: usize, out: *mut f32) {
    act::silu_mul(gate, up, len, out)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gk_lm_attention(
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
    causal: i32,
    out: *mut f32,
) -> i32 {
    attn::lm_attention(q, k, v, cos, sin, kt, vc, cap, past, t, causal != 0, out)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gk_vis_attention(
    qkv: *mut f32,
    q_norm: *const f32,
    k_norm: *const f32,
    eps: f32,
    cos: *const f32,
    sin: *const f32,
    n: usize,
    out: *mut f32,
) -> i32 {
    attn::vis_attention(qkv, q_norm, k_norm, eps, cos, sin, n, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_gray_chroma(rgb: *const u8, npix: usize, gray: *mut u8, chroma: *mut u8, hist: *mut u64) {
    image::gray_chroma(rgb, npix, gray, chroma, hist)
}

#[no_mangle]
pub unsafe extern "C" fn gk_threshold(src: *const u8, npix: usize, thr: u8, ge: i32, out: *mut u8) {
    image::threshold(src, npix, thr, ge != 0, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_dilate(mask: *const u8, h: usize, w: usize, r: usize, out: *mut u8) {
    image::dilate(mask, h, w, r, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_long_runs(mask: *const u8, h: usize, w: usize, min_len: usize, axis: u32, out: *mut u8) {
    image::long_runs(mask, h, w, min_len, axis, out)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gk_bridges(line: *const u8, text: *const u8, h: usize, w: usize, axis: u32, reach: usize, out: *mut u8) {
    image::bridges(line, text, h, w, axis, reach, out)
}

#[no_mangle]
pub unsafe extern "C" fn gk_damp(rgb: *mut u8, mask: *const u8, npix: usize, luma: u8) {
    image::damp(rgb, mask, npix, luma)
}

#[no_mangle]
pub unsafe extern "C" fn gk_label(mask: *const u8, h: usize, w: usize, labels: *mut i32, boxes: *mut i32, max_boxes: usize, counts: *mut i64) -> i64 {
    image::label(mask, h, w, labels, boxes, max_boxes, counts)
}

#[no_mangle]
pub unsafe extern "C" fn gk_select(labels: *const i32, npix: usize, flags: *const u8, nflags: usize, out: *mut u8) {
    image::select(labels, npix, flags, nflags, out)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gk_patchify(rgb: *const u8, gh: usize, gw: usize, inv255: f32, mean: *const f32, std: *const f32, out: *mut f32) {
    image::patchify(rgb, gh, gw, inv255, mean, std, out)
}
