//! GPU engine of the gravity-ocr runtime, loaded by `gravity_ocr/gpu.py` with `ctypes`.
//!
//! Built on the candle-rocm-vulkan fork: features `rocm` (AMD, with the HIP kernels of
//! `kernels/gravity.hip`), `vulkan` (any Vulkan 1.1 GPU) and `cuda` (NVIDIA).  The Python side
//! uploads the packed weights by name once, then runs whole vision / language passes; only
//! the inputs and the results cross the bus.  Every entry point returns a negative code on
//! failure and leaves the reason in `gg_last_error`; panics are caught at the boundary.

#![allow(clippy::missing_safety_doc)]

mod engine;
mod ggml;
#[cfg(feature = "rocm")]
mod hipk;

use engine::{Cache, Engine, Kind};
use gravity_kernels::codec::QTensor;
use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// ABI of the exported functions; `gravity_ocr/gpu.py` refuses a library whose version differs.
pub const VERSION: u32 = 2;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(e: impl std::fmt::Display) {
    LAST_ERROR.with(|s| *s.borrow_mut() = e.to_string());
}

/// Run `f`, turning errors and panics into `-1` and a message.
fn guard<F: FnOnce() -> candle_core::Result<i64>>(f: F) -> i64 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(e);
            -1
        }
        Err(p) => {
            let m = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
            set_error(format!("panic: {}", m.unwrap_or_default()));
            -2
        }
    }
}

unsafe fn slice<'a>(p: *const f32, n: usize) -> &'a [f32] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

unsafe fn write(out: *mut f32, v: &[f32]) {
    std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
}

#[no_mangle]
pub extern "C" fn gg_version() -> u32 {
    VERSION
}

/// Backends compiled in: bit 0 rocm, bit 1 vulkan, bit 2 cuda.
#[no_mangle]
pub extern "C" fn gg_backends() -> u32 {
    (cfg!(feature = "rocm") as u32) | ((cfg!(feature = "vulkan") as u32) << 1) | ((cfg!(feature = "cuda") as u32) << 2)
}

/// Number of usable devices of a backend (1 rocm, 2 vulkan, 3 cuda); 0 when none or not built.
#[no_mangle]
pub extern "C" fn gg_device_count(kind: u32) -> i64 {
    guard(|| {
        Ok(match kind {
            #[cfg(feature = "rocm")]
            1 => candle_core::rocm::device_count() as i64,
            #[cfg(feature = "vulkan")]
            2 => candle_core::vulkan_backend::device_count().unwrap_or(0) as i64,
            #[cfg(feature = "cuda")]
            3 => candle_core::utils::cuda_is_available() as i64,
            _ => 0,
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_last_error(buf: *mut c_char, cap: usize) -> usize {
    LAST_ERROR.with(|s| {
        let b = s.borrow();
        let n = b.len().min(cap.saturating_sub(1));
        if !buf.is_null() && cap > 0 {
            std::ptr::copy_nonoverlapping(b.as_ptr() as *const c_char, buf, n);
            *buf.add(n) = 0;
        }
        b.len()
    })
}

/// Open a device: kind 1 rocm, 2 vulkan, 3 cuda, 4 cpu.  Null on failure.
#[no_mangle]
pub extern "C" fn gg_open(kind: u32, ordinal: u32) -> *mut Engine {
    let mut out: *mut Engine = std::ptr::null_mut();
    let rc = guard(|| {
        let k = match kind {
            1 => Kind::Rocm,
            2 => Kind::Vulkan,
            3 => Kind::Cuda,
            4 => Kind::Cpu,
            _ => candle_core::bail!("unknown backend {kind}"),
        };
        out = Box::into_raw(Box::new(Engine::open(k, ordinal as usize)?));
        Ok(0)
    });
    if rc < 0 {
        return std::ptr::null_mut();
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn gg_close(e: *mut Engine) {
    if !e.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(e))));
    }
}

/// JSON description of the engine; returns its length (writes up to `cap - 1` bytes).
#[no_mangle]
pub unsafe extern "C" fn gg_describe(e: *const Engine, buf: *mut c_char, cap: usize) -> usize {
    let s = (*e).describe();
    let n = s.len().min(cap.saturating_sub(1));
    if !buf.is_null() && cap > 0 {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
    s.len()
}

/// Options: 1 gemv_max (rows), 2 custom kernels on/off, 3 fused attention on/off,
/// 4 qgemm_max (rows; negative: no limit), 5 vision attention (0 flash, 1 chunked),
/// 6 bytes of scores per vision chunk, 7 language attention (0 auto, 1 flash, 2 per-row kernel,
/// 3 tiled kernel; the kernels only for passes of up to 64 rows), 8 storage of the matrices
/// uploaded from now on (0 float32, 1 the bf16 / f16 of the source, 2 FP8, 3 NVFP4, 4 MXFP8,
/// 5 MXFP4), 9 matmul of half
/// matrices (0 widened to f32, 1 in the half type), 10 bytes of float32 rows widened or decoded
/// at a time, 11 format of new KV caches (0 float32, 1 FP8, 2 TurboQuant 4 bit, 3 3 bit,
/// 4 NVFP4, 5 MXFP8, 6 MXFP4).
#[no_mangle]
pub unsafe extern "C" fn gg_set_option(e: *mut Engine, key: u32, value: i64) -> i64 {
    let e = &mut *e;
    guard(|| {
        match key {
            1 => e.gemv_max = value.clamp(0, 32) as usize,
            2 => {
                #[cfg(feature = "rocm")]
                {
                    e.custom = value != 0 && e.hip.is_some();
                }
            }
            3 => e.flash = value != 0,
            4 => e.qgemm_max = if value < 0 { usize::MAX } else { value as usize },
            5 => e.vis_attn = value.clamp(0, 1) as u32,
            6 => e.chunk_bytes = value.max(1 << 20) as usize,
            7 => e.lm_attn = value.clamp(0, 3) as u32,
            8 => e.weight_store = value.clamp(0, 5) as u32,
            9 => e.half_compute = value.clamp(0, 1) as u32,
            10 => e.decode_bytes = value.max(1 << 20) as usize,
            11 => e.kv_format = value.clamp(0, 6) as u32,
            _ => candle_core::bail!("unknown option {key}"),
        }
        Ok(0)
    })
}

/// Return the memory the allocator keeps for reuse to the driver (weights stay).
#[no_mangle]
pub unsafe extern "C" fn gg_trim(e: *const Engine) -> i64 {
    let e = &*e;
    guard(|| {
        e.trim()?;
        Ok(0)
    })
}

/// Drop the weights whose names start with `prefix` and return the memory to the driver;
/// returns the bytes the weights took.
#[no_mangle]
pub unsafe extern "C" fn gg_drop(e: *mut Engine, prefix: *const c_char) -> i64 {
    let e = &mut *e;
    guard(|| {
        let p = CStr::from_ptr(prefix).to_str().map_err(|x| candle_core::Error::Msg(x.to_string()))?;
        Ok(e.drop_prefix(p)? as i64)
    })
}

/// Bytes of one row of `cols` values in GGML type `dtype` (negative: not whole blocks).
#[no_mangle]
pub extern "C" fn gg_ggml_row_bytes(dtype: u32, cols: usize) -> i64 {
    guard(|| Ok(ggml::row_bytes(ggml::dtype_of(dtype)?, cols)? as i64))
}

/// Quantise `t` into GGML `dtype` blocks on the CPU (`imatrix`: one importance weight per
/// column, or null); writes `out` (`cap` bytes) and returns the bytes written.
#[no_mangle]
pub unsafe extern "C" fn gg_ggml_quantize(t: *const QTensor, dtype: u32, imatrix: *const f32, out: *mut u8, cap: usize) -> i64 {
    guard(|| {
        let t = &*t;
        let w = if imatrix.is_null() { None } else { Some(std::slice::from_raw_parts(imatrix, t.cols as usize)) };
        let out = std::slice::from_raw_parts_mut(out, cap);
        Ok(ggml::quantize(t, ggml::dtype_of(dtype)?, w, out)? as i64)
    })
}

/// GGML `dtype` blocks of a `rows × cols` tensor (`len` bytes) as float32 into `out` (CPU).
#[no_mangle]
pub unsafe extern "C" fn gg_ggml_dequant(dtype: u32, rows: usize, cols: usize, data: *const u8, len: usize, out: *mut f32) -> i64 {
    guard(|| {
        let data = std::slice::from_raw_parts(data, len);
        let out = std::slice::from_raw_parts_mut(out, rows * cols);
        ggml::dequant(ggml::dtype_of(dtype)?, rows, cols, data, out)?;
        Ok(0)
    })
}

/// Upload GGML `dtype` blocks (`len` bytes, e.g. straight from a memory-mapped GGUF file) as
/// matrix `name` `[rows, cols]`, multiplied on the device without expanding it.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_set_ggml(e: *mut Engine, name: *const c_char, dtype: u32, rows: usize, cols: usize, data: *const u8, len: usize) -> i64 {
    let e = &mut *e;
    guard(|| {
        let name = CStr::from_ptr(name).to_str().map_err(|x| candle_core::Error::Msg(x.to_string()))?;
        e.set_ggml(name, dtype, rows, cols, std::slice::from_raw_parts(data, len))?;
        Ok(0)
    })
}

/// `x [n, cols] · Wᵀ` on the device with `W` as GGML `dtype` blocks; writes `out [n, rows]`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_test_ggml_linear(e: *const Engine, dtype: u32, rows: usize, cols: usize, data: *const u8, len: usize, x: *const f32, n: usize, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let v = e.test_ggml_linear(dtype, rows, cols, std::slice::from_raw_parts(data, len), slice(x, n * cols), n)?;
        write(out, &v);
        Ok(0)
    })
}

/// Tensors and metadata entries of a GGUF file as candle's reader sees them.
#[no_mangle]
pub unsafe extern "C" fn gg_gguf_info(path: *const c_char, tensors: *mut u64, kvs: *mut u64) -> i64 {
    guard(|| {
        let p = CStr::from_ptr(path).to_str().map_err(|x| candle_core::Error::Msg(x.to_string()))?;
        let (t, k) = ggml::gguf_info(p)?;
        *tensors = t as u64;
        *kvs = k as u64;
        Ok(0)
    })
}

/// Bytes the K/V tensors of a cache take (its capacity included).
#[no_mangle]
pub unsafe extern "C" fn gg_cache_bytes(c: *const Cache) -> u64 {
    (*c).bytes() as u64
}

/// Bytes of one cached position in KV `format` (0 float32, 1 FP8, 2 / 3 TurboQuant 4 / 3 bit).
#[no_mangle]
pub unsafe extern "C" fn gg_kv_row_bytes(e: *const Engine, format: u32) -> i64 {
    let e = &*e;
    guard(|| Ok(e.kv_row_bytes(format)? as i64))
}

/// Free the weights and return the device memory to the driver (before exit).
#[no_mangle]
pub unsafe extern "C" fn gg_release(e: *mut Engine) -> i64 {
    let e = &mut *e;
    guard(|| {
        e.release()?;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_mem(e: *const Engine, free: *mut u64, total: *mut u64) -> i64 {
    let e = &*e;
    guard(|| {
        let (f, t) = e.mem()?;
        *free = f as u64;
        *total = t as u64;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_weight_bytes(e: *const Engine) -> u64 {
    (*e).weight_bytes() as u64
}

#[no_mangle]
pub unsafe extern "C" fn gg_synchronize(e: *const Engine) -> i64 {
    let e = &*e;
    guard(|| {
        e.synchronize()?;
        Ok(0)
    })
}

/// Upload one tensor by name; `vector` for norms and biases (kept as float32).
#[no_mangle]
pub unsafe extern "C" fn gg_set_tensor(e: *mut Engine, name: *const c_char, t: *const QTensor, vector: i32) -> i64 {
    let e = &mut *e;
    guard(|| {
        let name = CStr::from_ptr(name).to_str().map_err(|x| candle_core::Error::Msg(x.to_string()))?;
        e.set(name, &*t, vector != 0)?;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_head_rows(e: *const Engine) -> i64 {
    (*e).head_rows() as i64
}

/// Vision tower: patches `[n, 588]`, cos / sin `[n, 64]` → `out [n/4, 1536]`.  With `nsegs > 0`
/// the rows are a batch of images of `segs[i]` patches each (attention stays inside an image).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_vision(e: *const Engine, patches: *const f32, n: usize, cos: *const f32, sin: *const f32, segs: *const u64, nsegs: usize, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let segs: Vec<usize> = if nsegs == 0 || segs.is_null() { Vec::new() } else { std::slice::from_raw_parts(segs, nsegs).iter().map(|&s| s as usize).collect() };
        let v = e.vision(slice(patches, n * engine::PATCH_ROW), n, slice(cos, n * 64), slice(sin, n * 64), &segs)?;
        write(out, &v);
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_cache_new(e: *const Engine, cap: usize) -> *mut Cache {
    let e = &*e;
    let mut out: *mut Cache = std::ptr::null_mut();
    let rc = guard(|| {
        out = Box::into_raw(Box::new(e.new_cache(cap)?));
        Ok(0)
    });
    if rc < 0 {
        return std::ptr::null_mut();
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn gg_cache_grow(e: *const Engine, c: *mut Cache, cap: usize, len: usize) -> i64 {
    let (e, c) = (&*e, &mut *c);
    guard(|| {
        e.grow_cache(c, cap, len)?;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn gg_cache_free(c: *mut Cache) {
    if !c.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(c))));
    }
}

/// Language model: `t` rows `x [t, 1536]` at rotary tables `[t, 64]` after `past` cached
/// rows; writes the logits of rows `from..t` (`[(t - from), head_rows]`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_lm(e: *const Engine, c: *const Cache, x: *const f32, t: usize, cos: *const f32, sin: *const f32, past: usize, causal: i32, from: usize, inject: *const f32, logits: *mut f32) -> i64 {
    gg_lm_rest(e, c, x, t, cos, sin, past, causal, from, inject, logits, std::ptr::null_mut())
}

/// `gg_lm` that also writes, for every logits row, the best of the output rows a sliced head
/// cut (tensor `lm_head.rest.weight`) as `[logit, row]` (row as f32) into `rest` (`[(t - from), 2]`);
/// a null `rest` skips that.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_lm_rest(e: *const Engine, c: *const Cache, x: *const f32, t: usize, cos: *const f32, sin: *const f32, past: usize, causal: i32, from: usize, inject: *const f32, logits: *mut f32, rest: *mut f32) -> i64 {
    let (e, c) = (&*e, &*c);
    guard(|| {
        let inj = if inject.is_null() { None } else { Some(slice(inject, t * engine::LM_WIDTH)) };
        let (v, best) = e.lm(c, slice(x, t * engine::LM_WIDTH), t, slice(cos, t * 64), slice(sin, t * 64), past, causal != 0, from, inj, !rest.is_null())?;
        write(logits, &v);
        if !rest.is_null() {
            write(rest, &best);
        }
        Ok(0)
    })
}

/// Evaluate the cut output rows (`lm_head.rest.weight`, e.g. Q4_K GGML in VRAM) for a batch of
/// Cauchy-Schwarz uncertain hidden states `x [n_rows, k_dim]`, writing the best logit into
/// `out_max [n_rows]` and its row index within `REST` into `out_id [n_rows]`.
#[no_mangle]
pub unsafe extern "C" fn gg_rest_max(
    e: *const Engine,
    x: *const f32,
    n_rows: usize,
    k_dim: usize,
    out_max: *mut f32,
    out_id: *mut u32,
) -> i64 {
    if e.is_null() || x.is_null() || out_max.is_null() || out_id.is_null() {
        return -1;
    }
    if n_rows == 0 {
        return 0;
    }
    let e = &*e;
    guard(|| {
        let (vals, rows) = e.eval_rest_max(slice(x, n_rows * k_dim), n_rows, k_dim)?;
        write(out_max, &vals);
        std::ptr::copy_nonoverlapping(rows.as_ptr(), out_id, rows.len());
        Ok(0)
    })
}

/// The step profile of the language passes so far (`GRAVITY_GPU_PROFILE=1`) as JSON into `buf`;
/// clears it.  Returns the length the JSON needs.
#[no_mangle]
pub unsafe extern "C" fn gg_profile(buf: *mut c_char, len: usize) -> usize {
    let s = engine::profile_take();
    if !buf.is_null() && len > 0 {
        let n = s.len().min(len - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    s.len()
}

// ------------------------------------------------------------------ self-check entry points
#[no_mangle]
pub unsafe extern "C" fn gg_test_dequant(e: *const Engine, t: *const QTensor, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let v = e.test_dequant(&*t)?;
        write(out, &v);
        Ok(0)
    })
}

/// Seconds per product (path 0 auto, 1 gemv, 2 tiled fused gemm, 3 decode + BLAS); negative on error.
#[no_mangle]
pub unsafe extern "C" fn gg_bench_linear(e: *const Engine, t: *const QTensor, n: usize, path: u32, reps: usize) -> f64 {
    let e = &*e;
    match catch_unwind(AssertUnwindSafe(|| e.bench_linear(&*t, n, path, reps))) {
        Ok(Ok(s)) => s,
        Ok(Err(x)) => {
            set_error(x);
            -1.0
        }
        Err(_) => {
            set_error("panic");
            -2.0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gg_test_linear(e: *const Engine, t: *const QTensor, x: *const f32, n: usize, bias: *const f32, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let t = &*t;
        let b = if bias.is_null() { None } else { Some(slice(bias, t.rows as usize)) };
        let v = e.test_linear(t, slice(x, n * t.cols as usize), n, b)?;
        write(out, &v);
        Ok(0)
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_test_lm_attention(
    e: *const Engine,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    cos: *const f32,
    sin: *const f32,
    kold: *const f32,
    vold: *const f32,
    past: usize,
    t: usize,
    causal: i32,
    out: *mut f32,
    kcache: *mut f32,
) -> i64 {
    let e = &*e;
    guard(|| {
        let kv = engine::LM_KV * engine::LM_DIM;
        let (o, kc) = e.test_lm_attention(
            slice(q, t * engine::LM_HEADS * engine::LM_DIM),
            slice(k, t * kv),
            slice(v, t * kv),
            slice(cos, t * 64),
            slice(sin, t * 64),
            slice(kold, past * kv),
            slice(vold, past * kv),
            past,
            t,
            causal != 0,
        )?;
        write(out, &o);
        write(kcache, &kc);
        Ok(0)
    })
}

/// [`gg_test_lm_attention`] through a KV cache in `format`; writes the attention output only.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_test_kv_attention(
    e: *const Engine,
    format: u32,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    cos: *const f32,
    sin: *const f32,
    kold: *const f32,
    vold: *const f32,
    past: usize,
    t: usize,
    causal: i32,
    out: *mut f32,
) -> i64 {
    let e = &*e;
    guard(|| {
        let kv = engine::LM_KV * engine::LM_DIM;
        let o = e.test_kv_attention(
            format,
            slice(q, t * engine::LM_HEADS * engine::LM_DIM),
            slice(k, t * kv),
            slice(v, t * kv),
            slice(cos, t * 64),
            slice(sin, t * 64),
            slice(kold, past * kv),
            slice(vold, past * kv),
            past,
            t,
            causal != 0,
        )?;
        write(out, &o);
        Ok(0)
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_test_vis_attention(e: *const Engine, qkv: *const f32, qn: *const f32, kn: *const f32, cos: *const f32, sin: *const f32, n: usize, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let v = e.test_vis_attention(slice(qkv, n * 3 * engine::VIS_WIDTH), slice(qn, engine::VIS_DIM), slice(kn, engine::VIS_DIM), slice(cos, n * 64), slice(sin, n * 64), n)?;
        write(out, &v);
        Ok(0)
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gg_test_rows(e: *const Engine, op: u32, x: *const f32, n: usize, d: usize, w: *const f32, b: *const f32, out: *mut f32) -> i64 {
    let e = &*e;
    guard(|| {
        let (xl, wl, bl) = match op {
            0 => (n * d, d, 0),
            1 => (n * d, d, d),
            2 => (n * 2 * d, 0, 0),
            _ => (n * d, n * d, 0),
        };
        let v = e.test_rows(op, slice(x, xl), n, d, slice(w, wl), slice(b, bl))?;
        write(out, &v);
        Ok(0)
    })
}
