//! The HIP kernels of `kernels/gravity.hip`, launched on the stream candle uses.

use candle_core::{bail, DType, Result, Storage, Tensor};
use hip_runtime::module::HipModule;
use std::ffi::c_void;

/// The code object `build.rs` compiled.
pub const IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gravity.co"));
pub const ARCHS: &str = env!("GRAVITY_HIP_ARCHS");

pub const SCHEME_NAMES: [&str; 9] = ["f32", "f16", "bf16", "q8", "q6", "q5", "q4", "q3", "nf4"];
pub const GEMV_SIZES: [usize; 6] = [1, 2, 4, 8, 16, 32];
const LANES: u32 = 32;
const ROWS_PER_BLOCK: u32 = 8;

/// Packed parts on the device, in the layout of `quant.py`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QW {
    pub q: *const c_void,
    pub d: *const c_void,
    pub s: *const c_void,
    pub m: *const c_void,
    pub dm: *const c_void,
}

pub struct Kernels {
    module: HipModule,
}

fn hip<T>(r: std::result::Result<T, hip_runtime::error::HipError>) -> Result<T> {
    r.map_err(|e| candle_core::Error::Msg(format!("hip: {e}")))
}

/// Device address of the first element of `t`.
pub fn ptr(t: &Tensor) -> Result<*mut c_void> {
    let (st, layout) = t.storage_and_layout();
    match &*st {
        Storage::Rocm(s) => {
            let off = layout.start_offset() * t.dtype().size_in_bytes();
            Ok(unsafe { (s.device_ptr() as *mut u8).add(off) } as *mut c_void)
        }
        _ => bail!("gravity_gpu: tensor is not on a ROCm device"),
    }
}

fn opt_ptr(t: Option<&Tensor>) -> Result<*const c_void> {
    Ok(match t {
        Some(t) => ptr(t)? as *const c_void,
        None => std::ptr::null(),
    })
}

fn grid_1d(n: usize, block: u32) -> u32 {
    (n.div_ceil(block as usize)).clamp(1, 65535 * 16) as u32
}

impl Kernels {
    pub fn load() -> Result<Self> {
        if IMAGE.is_empty() {
            bail!("gravity_gpu: the HIP kernels were not compiled into this build");
        }
        let module = hip(HipModule::load_data(IMAGE))?;
        // a missing function means the image does not match this source
        hip(module.get_function("gk_gemv_q4_1"))?;
        Ok(Self { module })
    }

    unsafe fn launch(&self, name: &str, grid: u32, block: u32, args: &mut [*mut c_void]) -> Result<()> {
        let f = hip(self.module.get_function(name))?;
        hip(HipModule::launch(f, (grid, 1, 1), (block, 1, 1), 0, args))
    }

    pub fn dequant(&self, scheme: u32, w: QW, out: &Tensor, n: usize) -> Result<()> {
        let name = format!("gk_dequant_{}", SCHEME_NAMES[scheme as usize]);
        let mut w = w;
        let mut o = ptr(out)?;
        let mut n = n;
        let mut args = [&mut w as *mut QW as *mut c_void, &mut o as *mut _ as *mut c_void, &mut n as *mut usize as *mut c_void];
        unsafe { self.launch(&name, grid_1d(n, 256).min(65535), 256, &mut args) }
    }

    /// `y[n, rows] (+)= x[n, cols] · Wᵀ + bias` for `n <= 32`, `cols % 256 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(&self, scheme: u32, w: QW, x: &Tensor, bias: Option<&Tensor>, y: &Tensor, rows: usize, cols: usize, n: usize, accumulate: bool) -> Result<()> {
        let nx = *GEMV_SIZES.iter().find(|&&s| s >= n).ok_or_else(|| candle_core::Error::Msg("gemv: too many rows".into()))?;
        let name = format!("gk_gemv_{}_{}", SCHEME_NAMES[scheme as usize], nx);
        let mut w = w;
        let mut xp = ptr(x)? as *const c_void;
        let mut bp = opt_ptr(bias)?;
        let mut yp = ptr(y)?;
        let (mut r, mut c, mut nn, mut acc) = (rows as i32, cols as i32, n as i32, accumulate as i32);
        let mut args = [
            &mut w as *mut QW as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut r as *mut i32 as *mut c_void,
            &mut c as *mut i32 as *mut c_void,
            &mut nn as *mut i32 as *mut c_void,
            &mut acc as *mut i32 as *mut c_void,
        ];
        let grid = (rows as u32).div_ceil(ROWS_PER_BLOCK);
        unsafe { self.launch(&name, grid, LANES * ROWS_PER_BLOCK, &mut args) }
    }

    /// `y[n, rows] (+)= x[n, cols] · Wᵀ + bias`, tiled in shared memory, `cols % 256 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn qgemm(&self, scheme: u32, w: QW, x: &Tensor, bias: Option<&Tensor>, y: &Tensor, rows: usize, cols: usize, n: usize, accumulate: bool) -> Result<()> {
        let bm = if n <= 32 { 32 } else { 64 };
        let name = format!("gk_qgemm_{}_{}", SCHEME_NAMES[scheme as usize], bm);
        let mut w = w;
        let mut xp = ptr(x)? as *const c_void;
        let mut bp = opt_ptr(bias)?;
        let mut yp = ptr(y)?;
        let (mut r, mut c, mut nn, mut acc) = (rows as i32, cols as i32, n as i32, accumulate as i32);
        let mut args = [
            &mut w as *mut QW as *mut c_void,
            &mut xp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut r as *mut i32 as *mut c_void,
            &mut c as *mut i32 as *mut c_void,
            &mut nn as *mut i32 as *mut c_void,
            &mut acc as *mut i32 as *mut c_void,
        ];
        let f = hip(self.module.get_function(&name))?;
        let grid = ((rows as u32).div_ceil(64), (n as u32).div_ceil(bm), 1);
        unsafe { hip(HipModule::launch(f, grid, (256, 1, 1), 0, &mut args)) }
    }

    /// Rotate q `[t, 16, 128]` in place, rotate k and write k / v `[t, 8, 128]` to the caches
    /// `[cap, 8, 128]` at row `past`.
    #[allow(clippy::too_many_arguments)]
    pub fn lm_rope_cache(&self, q: &Tensor, k: &Tensor, v: &Tensor, cos: &Tensor, sin: &Tensor, kc: &Tensor, vc: &Tensor, t: usize, past: usize) -> Result<()> {
        for x in [q, k, v, cos, sin] {
            if !x.is_contiguous() || x.dtype() != DType::F32 {
                bail!("lm_rope_cache: inputs must be contiguous f32");
            }
        }
        let mut ps = [ptr(q)?, ptr(k)?, ptr(v)?, ptr(cos)?, ptr(sin)?, ptr(kc)?, ptr(vc)?];
        let (mut tt, mut pp) = (t as i32, past as i32);
        let mut args: Vec<*mut c_void> = ps.iter_mut().map(|p| p as *mut _ as *mut c_void).collect();
        args.push(&mut tt as *mut i32 as *mut c_void);
        args.push(&mut pp as *mut i32 as *mut c_void);
        unsafe { self.launch("gk_lm_rope_cache", (t * 32) as u32, 64, &mut args) }
    }

    /// Grouped-query attention of `t` query rows `q [t, 16, 128]` over the cache rows written so far
    /// (`kc` / `vc [cap, 8, 128]`, `past + t` rows); causal = bottom-right aligned.
    /// `tiled`: one block per (KV head, 8 query rows) sharing each loaded chunk of keys, else one
    /// block per (query row, head).
    #[allow(clippy::too_many_arguments)]
    pub fn lm_attention(&self, q: &Tensor, kc: &Tensor, vc: &Tensor, out: &Tensor, t: usize, past: usize, causal: bool, scale: f32, tiled: bool) -> Result<()> {
        for x in [q, kc, vc, out] {
            if !x.is_contiguous() || x.dtype() != DType::F32 {
                bail!("lm_attention: tensors must be contiguous f32");
            }
        }
        let mut ps = [ptr(q)?, ptr(kc)?, ptr(vc)?, ptr(out)?];
        let (mut tt, mut pp, mut cc, mut ss) = (t as i32, past as i32, causal as i32, scale);
        let mut args: Vec<*mut c_void> = ps.iter_mut().map(|p| p as *mut _ as *mut c_void).collect();
        args.push(&mut tt as *mut i32 as *mut c_void);
        args.push(&mut pp as *mut i32 as *mut c_void);
        args.push(&mut cc as *mut i32 as *mut c_void);
        args.push(&mut ss as *mut f32 as *mut c_void);
        if tiled {
            unsafe { self.launch("gk_lm_attention_tiled", (t.div_ceil(8) * 8) as u32, 128, &mut args) }
        } else {
            unsafe { self.launch("gk_lm_attention", (t * 16) as u32, 128, &mut args) }
        }
    }

    /// qkv `[n, 3072]` → q, k (normalised, rotated), v, each `[n, 16, 64]`.
    #[allow(clippy::too_many_arguments)]
    pub fn vis_qkv(&self, qkv: &Tensor, qn: &Tensor, kn: &Tensor, cos: &Tensor, sin: &Tensor, eps: f32, q: &Tensor, k: &Tensor, v: &Tensor, n: usize) -> Result<()> {
        let mut ps = [ptr(qkv)?, ptr(qn)?, ptr(kn)?, ptr(cos)?, ptr(sin)?];
        let mut e = eps;
        let mut outs = [ptr(q)?, ptr(k)?, ptr(v)?];
        let mut nn = n as i32;
        let mut args: Vec<*mut c_void> = ps.iter_mut().map(|p| p as *mut _ as *mut c_void).collect();
        args.push(&mut e as *mut f32 as *mut c_void);
        for p in outs.iter_mut() {
            args.push(p as *mut _ as *mut c_void);
        }
        args.push(&mut nn as *mut i32 as *mut c_void);
        unsafe { self.launch("gk_vis_qkv", (n * 48) as u32, 64, &mut args) }
    }

    /// `x[n, d] += rms_norm(y) · w` in place.
    pub fn rms_norm_add(&self, x: &Tensor, y: &Tensor, w: &Tensor, eps: f32, n: usize, d: usize) -> Result<()> {
        let (mut xp, mut yp, mut wp) = (ptr(x)?, ptr(y)?, ptr(w)?);
        let (mut e, mut dd) = (eps, d as i32);
        let mut args = [
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut e as *mut f32 as *mut c_void,
            &mut dd as *mut i32 as *mut c_void,
        ];
        unsafe { self.launch("gk_rms_norm_add", n as u32, 256, &mut args) }
    }

    pub fn swiglu_split(&self, u: &Tensor, out: &Tensor, n: usize, inter: usize) -> Result<()> {
        let (mut up, mut op) = (ptr(u)?, ptr(out)?);
        let (mut nn, mut ii) = (n, inter);
        let mut args = [
            &mut up as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nn as *mut usize as *mut c_void,
            &mut ii as *mut usize as *mut c_void,
        ];
        unsafe { self.launch("gk_swiglu_split", grid_1d(n * inter, 256).min(65535), 256, &mut args) }
    }

    pub fn silu_mul(&self, g: &Tensor, u: &Tensor, out: &Tensor, len: usize) -> Result<()> {
        let (mut gp, mut up, mut op) = (ptr(g)?, ptr(u)?, ptr(out)?);
        let mut l = len;
        let mut args = [
            &mut gp as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut l as *mut usize as *mut c_void,
        ];
        unsafe { self.launch("gk_silu_mul", grid_1d(len, 256).min(65535), 256, &mut args) }
    }
}
