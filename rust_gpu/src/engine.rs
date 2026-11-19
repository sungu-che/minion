//! The model forward passes on a candle device.
//!
//! Two implementations of the fused operations share one forward:
//!
//! * `custom` (ROCm): the HIP kernels of `kernels/gravity.hip` — weights stay packed in VRAM
//!   and are decoded in registers by the GEMV kernel (few rows) or into a scratch matrix
//!   for the GEMM (many rows); rotary embedding + cache write, q/k norm + rotation and the
//!   gated activations are single kernels;
//! * portable (Vulkan, CUDA, or ROCm without the kernels): the same steps with candle ops,
//!   weights decoded once on the host (bit-exact codec of `gravity_kernels`) and kept as
//!   dense float32.
//!
//! Attention is candle-nn's fused flash attention (grouped-query, bottom-right causal).
//!
//! Memory (portable path, `weight_store`): a matrix is kept as float32 (0), as the bf16 / f16
//! values of the source (1, exact: widening is lossless; matmul by row chunks widened to f32,
//! or natively in the half type with `half_compute`), or as FP8 E4M3 rows with one scale per
//! segment of at most 1024 values (2, candle-nn TurboQuant `Fp8` codec; decoded by row chunks).
//! The KV cache (`kv_format`) is float32, FP8 or TurboQuant MSE codes; attention then runs on
//! the packed rows (`candle_nn::turbo_quant::attention_packed`, fused kernels when compiled in).
//! A group of weights can be dropped by name prefix (`drop_prefix`) and the device memory pool
//! handed back to the driver (`settle`), so the vision tower need not stay next to the
//! language model.

use candle_core::accel::{QuantFormat, RotationKind};
use candle_core::quantized::QMatMul;
use candle_core::{bail, DType, Device, Module, Result, Tensor};
use candle_nn::flash_attn::FlashAttnConfig;
use candle_nn::turbo_quant::TurboQuantizer;
use gravity_kernels::codec::{self, QTensor};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// `GRAVITY_GPU_PROFILE=1`: the language pass synchronises after every step and adds up the
/// time of each kind of step (`profile_take`).  Off, it costs one branch per step.
fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GRAVITY_GPU_PROFILE").map(|v| v == "1").unwrap_or(false))
}

static TIMES: Mutex<BTreeMap<&'static str, (f64, u64)>> = Mutex::new(BTreeMap::new());

/// The profile so far as JSON `{step: [seconds, count]}`; clears it.
pub fn profile_take() -> String {
    let mut m = TIMES.lock().unwrap();
    let body: Vec<String> = m.iter().map(|(k, (s, n))| format!("\"{k}\": [{s:.6}, {n}]")).collect();
    m.clear();
    format!("{{{}}}", body.join(", "))
}

pub const VIS_WIDTH: usize = 1024;
pub const VIS_HEADS: usize = 16;
pub const VIS_DIM: usize = 64;
pub const VIS_DEPTH: usize = 24;
pub const PATCH_ROW: usize = 588;
pub const LM_WIDTH: usize = 1536;
pub const LM_DEPTH: usize = 16;
pub const LM_HEADS: usize = 16;
pub const LM_KV: usize = 8;
pub const LM_DIM: usize = 128;
pub const LM_INTER: usize = 4608;
pub const EPS: f32 = 1e-5;
/// rows of a language pass up to which the custom attention kernels run (a prefill takes flash)
pub const LM_ATTN_ROWS: usize = 64;
/// rows from which the tiled kernel replaces the one-block-per-row kernel (`lm_attn` auto); RX 6600,
/// cache of 600 rows: per-row 2.0 / 4.5 / 7.4 / 13.6 ms and tiled 5.9 / 6.9 / 7.1 / 7.3 ms for the 16
/// layers at 1 / 8 / 16 / 32 rows (flash 4.4 / 13.3 / 15.2 / 15.7 ms)
pub const LM_ATTN_TILED_FROM: usize = 17;

const VIS: &str = "model.visual.";
const LM: &str = "model.language_model.";
pub const HEAD: &str = "lm_head.weight";
/// The output rows a sliced head cut (`gravity_ocr/gpu.py` REST).
pub const REST: &str = "lm_head.rest.weight";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Rocm,
    Vulkan,
    Cuda,
    Cpu,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Rocm => "rocm",
            Kind::Vulkan => "vulkan",
            Kind::Cuda => "cuda",
            Kind::Cpu => "cpu",
        }
    }
}

/// Storage of the matrices (`weight_store`): float32, the source's bf16 / f16, or rows in one
/// of the candle-nn TurboQuant codecs (FP8 E4M3 with a scale per segment, NVIDIA NVFP4: E2M1
/// with an E4M3 scale per 16 values, OCP MXFP8 / MXFP4: E8M0 scale per 32 values).
pub const STORE_F32: u32 = 0;
pub const STORE_HALF: u32 = 1;
pub const STORE_FP8: u32 = 2;
pub const STORE_NVFP4: u32 = 3;
pub const STORE_MXFP8: u32 = 4;
pub const STORE_MXFP4: u32 = 5;

/// KV cache formats (`kv_format`).
pub const KV_F32: u32 = 0;
pub const KV_FP8: u32 = 1;
pub const KV_TURBO4: u32 = 2;
pub const KV_TURBO3: u32 = 3;
pub const KV_NVFP4: u32 = 4;
pub const KV_MXFP8: u32 = 5;
pub const KV_MXFP4: u32 = 6;

/// The codec of a KV cache format (`None`: float32).  Keys and values are rotated first
/// (randomised Hadamard: incoherent processing before the FP4 / FP8 rounding).
fn kv_quant_format(kv: u32) -> Option<QuantFormat> {
    match kv {
        KV_FP8 => Some(QuantFormat::Fp8),
        KV_TURBO4 => Some(QuantFormat::TurboMse { bits: 4 }),
        KV_TURBO3 => Some(QuantFormat::TurboMse { bits: 3 }),
        KV_NVFP4 => Some(QuantFormat::Nvfp4),
        KV_MXFP8 => Some(QuantFormat::Mxfp8),
        KV_MXFP4 => Some(QuantFormat::Mxfp4),
        _ => None,
    }
}

fn kv_name(kv: u32) -> &'static str {
    match kv {
        KV_FP8 => "fp8",
        KV_TURBO4 => "turbo4",
        KV_TURBO3 => "turbo3",
        KV_NVFP4 => "nvfp4",
        KV_MXFP8 => "mxfp8",
        KV_MXFP4 => "mxfp4",
        _ => "f32",
    }
}

/// The runtime's name of a codec (`fp8`, `nvfp4`, `mxfp8`, `mxfp4`, `turbo4` …).
fn format_name(f: QuantFormat) -> String {
    match f {
        QuantFormat::Fp8 => "fp8".into(),
        QuantFormat::Nvfp4 => "nvfp4".into(),
        QuantFormat::Mxfp8 => "mxfp8".into(),
        QuantFormat::Mxfp4 => "mxfp4".into(),
        QuantFormat::TurboMse { bits } => format!("turbo{bits}"),
        QuantFormat::TurboProd { bits } => format!("prod{bits}"),
    }
}

/// The longest row the TurboQuant / FP8 codecs take (`candle_core::accel::formats::MAX_ROW_DIM`).
const CODEC_MAX_ROW: usize = 1024;

/// The codec of a weight store (`None` for float32 / half).
fn store_format(store: u32) -> Option<QuantFormat> {
    match store {
        STORE_FP8 => Some(QuantFormat::Fp8),
        STORE_NVFP4 => Some(QuantFormat::Nvfp4),
        STORE_MXFP8 => Some(QuantFormat::Mxfp8),
        STORE_MXFP4 => Some(QuantFormat::Mxfp4),
        _ => None,
    }
}

fn store_name(store: u32) -> &'static str {
    match store {
        STORE_HALF => "half",
        STORE_FP8 => "fp8",
        STORE_NVFP4 => "nvfp4",
        STORE_MXFP8 => "mxfp8",
        STORE_MXFP4 => "mxfp4",
        _ => "f32",
    }
}

/// Segment length of coded weight rows: the largest divisor of `cols` up to 1024 that holds
/// whole scale blocks of `format` (16 values NVFP4, 32 MX), at least 64 unless the row is
/// shorter.  Each segment is one codec row (FP8: one scale per segment).
fn codec_segment(cols: usize, format: QuantFormat) -> Option<usize> {
    let block = format.block_size().max(1);
    (1..=cols.min(CODEC_MAX_ROW)).rev().find(|s| cols % s == 0 && s % block == 0 && (*s >= 64 || *s == cols))
}

/// A weight matrix on the device.
pub enum Weight {
    /// float32 `[rows, cols]`
    Dense(Tensor),
    /// bf16 / f16 `[rows, cols]`: the source values as they are
    Half(Tensor),
    /// codec rows `[rows · cols/seg, words]` u32 (FP8 / NVFP4 / MXFP8 / MXFP4), `seg` values each
    Coded { rows: usize, cols: usize, data: Tensor, q: Arc<TurboQuantizer> },
    /// GGML blocks (Q8_0, Q4_K … from a GGUF file), multiplied as they are (`QMatMul`)
    Ggml { rows: usize, cols: usize, mm: QMatMul, bytes: usize },
    /// the packed parts as bytes (custom kernels only)
    #[allow(dead_code)]
    Packed { scheme: u32, rows: usize, cols: usize, parts: [Option<Tensor>; 5] },
}

impl Weight {
    pub fn bytes(&self) -> usize {
        match self {
            Weight::Dense(t) => t.elem_count() * 4,
            Weight::Half(t) => t.elem_count() * 2,
            Weight::Coded { data, .. } => data.elem_count() * 4,
            Weight::Ggml { bytes, .. } => *bytes,
            Weight::Packed { parts, .. } => parts.iter().flatten().map(|t| t.elem_count()).sum(),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Weight::Dense(t) | Weight::Half(t) => t.dims()[0],
            Weight::Coded { rows, .. } | Weight::Ggml { rows, .. } | Weight::Packed { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Weight::Dense(t) | Weight::Half(t) => t.dims()[1],
            Weight::Coded { cols, .. } | Weight::Ggml { cols, .. } | Weight::Packed { cols, .. } => *cols,
        }
    }
}

pub struct Cache {
    /// per layer: float32 `[cap, 8, 128]`, or packed u32 `[1, 8, cap, words]` with `quant`
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub cap: usize,
    /// quantizers (keys, values) of a compressed cache
    pub quant: Option<Arc<(TurboQuantizer, TurboQuantizer)>>,
}

impl Cache {
    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(|t| t.elem_count() * t.dtype().size_in_bytes()).sum()
    }
}

pub struct Engine {
    pub dev: Device,
    pub kind: Kind,
    #[cfg(feature = "rocm")]
    pub hip: Option<std::sync::Arc<crate::hipk::Kernels>>,
    pub mats: HashMap<String, Weight>,
    pub vecs: HashMap<String, Tensor>,
    /// rows of x up to which the packed GEMV runs (custom path)
    pub gemv_max: usize,
    /// rows of x up to which the tiled fused GEMM runs; above it: decode to scratch + BLAS
    pub qgemm_max: usize,
    /// use the custom kernels where they exist
    pub custom: bool,
    /// run attention through the fused flash kernels (else matmul + softmax)
    pub flash: bool,
    /// vision attention: 0 fused flash, 1 query chunks of BLAS matmul + fused softmax
    pub vis_attn: u32,
    /// language attention: 0 auto (custom kernels for a decoding pass on ROCm), 1 candle's flash
    /// attention, 2 / 3 the one-block-per-row / tiled kernel for a decoding pass
    pub lm_attn: u32,
    /// bytes one chunk of vision scores may take
    pub chunk_bytes: usize,
    /// how `set` stores matrices (`STORE_*`); the custom ROCm packed path wins where it applies
    pub weight_store: u32,
    /// half-stored matrices: 0 widen row chunks to f32 (bit-exact), 1 matmul in bf16 / f16
    pub half_compute: u32,
    /// bytes of the float32 rows widened / decoded at a time from a half or FP8 matrix
    pub decode_bytes: usize,
    /// format of new KV caches (`KV_*`)
    pub kv_format: u32,
    /// weight codec quantizers per (format, segment length)
    codecq: Mutex<HashMap<(u32, usize), Arc<TurboQuantizer>>>,
    pub note: String,
}

fn msg<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

impl Engine {
    pub fn open(kind: Kind, ordinal: usize) -> Result<Self> {
        let dev = match kind {
            #[cfg(feature = "rocm")]
            Kind::Rocm => Device::new_rocm(ordinal)?,
            #[cfg(feature = "vulkan")]
            Kind::Vulkan => Device::new_vulkan(ordinal)?,
            #[cfg(feature = "cuda")]
            Kind::Cuda => Device::new_cuda(ordinal)?,
            Kind::Cpu => Device::Cpu,
            #[allow(unreachable_patterns)]
            k => bail!("gravity_gpu was built without the {} backend", k.name()),
        };
        let mut e = Engine {
            dev,
            kind,
            #[cfg(feature = "rocm")]
            hip: None,
            mats: HashMap::new(),
            vecs: HashMap::new(),
            // measured on an RX 6600 (tools/bench_kernels.py): the packed GEMV wins up to 16 rows,
            // the tiled fused GEMM up to a few hundred, the decode + BLAS GEMM beyond
            gemv_max: 16,
            qgemm_max: 256,
            custom: false,
            vis_attn: 1,
            lm_attn: 0,
            chunk_bytes: 256 << 20,
            flash: true,
            weight_store: STORE_F32,
            half_compute: 0,
            decode_bytes: 64 << 20,
            kv_format: KV_F32,
            codecq: Mutex::new(HashMap::new()),
            note: String::new(),
        };
        #[cfg(feature = "rocm")]
        if kind == Kind::Rocm {
            match crate::hipk::Kernels::load() {
                Ok(k) => {
                    e.hip = Some(std::sync::Arc::new(k));
                    e.custom = true;
                }
                Err(err) => e.note = format!("custom HIP kernels unavailable ({err}); portable path"),
            }
        }
        Ok(e)
    }

    pub fn describe(&self) -> String {
        let qmax = if self.qgemm_max == usize::MAX { -1 } else { self.qgemm_max as i64 };
        let mut s = format!(
            "{{\"backend\":\"{}\",\"custom\":{},\"flash\":{},\"vis_attn\":{},\"lm_attn\":{},\"gemv_max\":{},\"qgemm_max\":{}",
            self.kind.name(),
            self.custom,
            self.flash,
            if self.vis_attn == 1 { "\"chunked\"" } else { "\"flash\"" },
            match (self.custom, self.lm_attn) {
                (false, _) | (_, 1) => "\"flash\"",
                (_, 2) => "\"kernel-rows\"",
                (_, 3) => "\"kernel-tiled\"",
                _ => "\"kernel-auto\"",
            },
            self.gemv_max,
            qmax
        );
        #[cfg(feature = "rocm")]
        if self.kind == Kind::Rocm {
            if let Ok(name) = candle_core::rocm::device_name(0) {
                s += &format!(",\"device\":{name:?}");
            }
            if let Ok(arch) = candle_core::rocm::device_arch(0) {
                s += &format!(",\"arch\":{arch:?}");
            }
            s += &format!(",\"kernel_archs\":{:?}", crate::hipk::ARCHS);
        }
        #[cfg(feature = "vulkan")]
        if self.kind == Kind::Vulkan {
            if let Ok(v) = self.dev.as_vulkan_device() {
                s += &format!(",\"device\":{:?},\"type\":{:?}", v.name(), v.device_type());
            }
        }
        s += &format!(
            ",\"weight_store\":\"{}\",\"half_compute\":{},\"kv_format\":\"{}\",\"flash_compiled\":{},\"turbo_compiled\":{}",
            store_name(self.weight_store),
            self.half_compute,
            kv_name(self.kv_format),
            candle_core::accel::config::FLASH_ATTN_COMPILED,
            candle_core::accel::config::TURBO_QUANT_COMPILED
        );
        // what the GPU does natively (candle accel profile): FP8 from sm_89, FP4 / MX from sm_100
        let p = candle_core::accel::profile(&self.dev);
        s += &format!(
            ",\"profile\":{{\"arch\":{:?},\"native_fp8\":{},\"native_fp4\":{},\"native_mx\":{},\"fp8_format\":\"{}\",\"fp4_format\":\"{}\"}}",
            p.arch,
            p.native_fp8,
            p.native_fp4,
            p.native_mx,
            format_name(p.fp8_format),
            format_name(p.fp4_format)
        );
        if !self.note.is_empty() {
            s += &format!(",\"note\":{:?}", self.note);
        }
        s + "}"
    }

    /// `(free, total)` device memory: the driver's free memory plus what the stream-ordered
    /// pool holds unused (this process can take it without waiting).  `(0, 0)` on the CPU.
    pub fn mem(&self) -> Result<(usize, usize)> {
        match self.dev.mem_info()? {
            Some((free, total)) => Ok((free.saturating_add(self.dev.pool_idle_bytes().unwrap_or(0)), total)),
            None => Ok((0, 0)),
        }
    }

    pub fn weight_bytes(&self) -> usize {
        self.mats.values().map(|w| w.bytes()).sum::<usize>() + self.vecs.values().map(|t| t.elem_count() * 4).sum::<usize>()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize()
    }

    /// Wait for the device, free the matmul workspaces and give the memory the pool keeps for
    /// reuse back to the driver (weights stay) – the order of blueprint-sandbox `settle`.
    pub fn settle(&self) -> Result<()> {
        if self.dev.is_cpu() {
            return Ok(());
        }
        self.dev.synchronize_all()?;
        self.dev.release_workspaces()?;
        #[cfg(feature = "rocm")]
        if self.kind == Kind::Rocm {
            self.dev.as_rocm_device()?.release_cached_resources()?;
        }
        self.dev.trim_memory_pool(0)
    }

    /// Give the memory the allocator keeps for reuse back to the driver (weights stay).
    pub fn trim(&self) -> Result<()> {
        self.settle()
    }

    /// Drop the weights whose names start with `prefix` and settle; returns the bytes freed.
    pub fn drop_prefix(&mut self, prefix: &str) -> Result<usize> {
        let before = self.weight_bytes();
        self.mats.retain(|k, _| !k.starts_with(prefix));
        self.vecs.retain(|k, _| !k.starts_with(prefix));
        let freed = before - self.weight_bytes();
        self.settle()?;
        Ok(freed)
    }

    /// Drop every weight and hand the device memory pool back to the driver.  Called before
    /// the process exits: the HIP runtime of Windows can hang tearing down a stream-ordered
    /// pool that still holds memory.
    pub fn release(&mut self) -> Result<()> {
        self.mats.clear();
        self.vecs.clear();
        self.codecq.lock().unwrap().clear();
        self.settle()
    }

    // --------------------------------------------------------------------------- weights
    fn host_f32(t: &QTensor) -> Result<Vec<f32>> {
        t.check().map_err(|c| msg(format!("tensor parts do not match the shape (code {c})")))?;
        let mut out = vec![0f32; t.numel()];
        unsafe { codec::dequant(t, out.as_mut_ptr()) };
        Ok(out)
    }

    /// Store a tensor: vectors (one row) as float32, matrices packed (custom path) or dense.
    pub fn set(&mut self, name: &str, t: &QTensor, vector: bool) -> Result<()> {
        let rows = t.rows as usize;
        let cols = t.cols as usize;
        if vector {
            let v = Self::host_f32(t)?;
            let n = v.len();
            self.vecs.insert(name.to_string(), Tensor::from_vec(v, n, &self.dev)?);
            return Ok(());
        }
        if self.custom && t.scheme != codec::F32 && t.rows_are_supers() {
            t.check().map_err(|c| msg(format!("{name}: parts do not match the shape (code {c})")))?;
            let up = |p: *const u8, len: u64| -> Result<Option<Tensor>> {
                if p.is_null() || len == 0 {
                    return Ok(None);
                }
                let bytes = unsafe { std::slice::from_raw_parts(p, len as usize) };
                // pad to a multiple of 16 bytes so vector loads never leave the buffer
                let mut v = bytes.to_vec();
                v.resize(len as usize + 16, 0);
                let n = v.len();
                Ok(Some(Tensor::from_vec(v, n, &self.dev)?))
            };
            let parts = [up(t.q, t.q_len)?, up(t.d, t.d_len)?, up(t.s, t.s_len)?, up(t.m, t.m_len)?, up(t.dm, t.dm_len)?];
            self.mats.insert(name.to_string(), Weight::Packed { scheme: t.scheme, rows, cols, parts });
            return Ok(());
        }
        // replace an old copy first: its memory is free again before the new one is allocated
        self.mats.remove(name);
        let coded = store_format(self.weight_store).filter(|f| codec_segment(cols, *f).is_some());
        let w = match coded {
            Some(f) => self.coded_weight(t, rows, cols, f)?,
            // a coded store whose rows do not split into its blocks keeps the source's half
            None if self.weight_store != STORE_F32 && Self::half_dtype(t).is_some() => Weight::Half(self.half_tensor(t, 0, rows)?),
            None => Weight::Dense(self.f32_weight(t, rows, cols)?),
        };
        self.mats.insert(name.to_string(), w);
        Ok(())
    }

    /// The dtype of a bf16 / f16 source, `None` for the others.
    fn half_dtype(t: &QTensor) -> Option<DType> {
        match t.scheme {
            codec::BF16 => Some(DType::BF16),
            codec::F16 => Some(DType::F16),
            _ => None,
        }
    }

    /// Rows `r0..r0 + n` of a bf16 / f16 source as they lie in the file, on the device.
    fn half_tensor(&self, t: &QTensor, r0: usize, n: usize) -> Result<Tensor> {
        let dt = Self::half_dtype(t).ok_or_else(|| msg("not a half tensor"))?;
        t.check().map_err(|c| msg(format!("tensor parts do not match the shape (code {c})")))?;
        let cols = t.cols as usize;
        let bytes = unsafe { std::slice::from_raw_parts(t.q.add(r0 * cols * 2), n * cols * 2) };
        Tensor::from_raw_buffer(bytes, dt, &[n, cols], &self.dev)
    }

    /// Rows of the source per step of a conversion: at most `decode_bytes` of float32.
    fn chunk_rows(&self, cols: usize) -> usize {
        (self.decode_bytes / (cols * 4)).max(1)
    }

    /// Float32 rows `r0..r0 + n` of any source on the device: a half source uploaded as it is and
    /// widened there, a packed one decoded on the host for these rows only (`codec::dequant_range`).
    fn f32_rows(&self, t: &QTensor, r0: usize, n: usize) -> Result<Tensor> {
        if Self::half_dtype(t).is_some() {
            return self.half_tensor(t, r0, n)?.to_dtype(DType::F32);
        }
        t.check().map_err(|c| msg(format!("tensor parts do not match the shape (code {c})")))?;
        let cols = t.cols as usize;
        let mut v = vec![0f32; n * cols];
        unsafe { codec::dequant_range(t, r0 * cols, n * cols, v.as_mut_ptr()) };
        Tensor::from_vec(v, (n, cols), &self.dev)
    }

    /// Float32 `[rows, cols]` on the device, converted by row chunks of at most `decode_bytes`:
    /// no float32 copy of the whole matrix in host memory (the 59,283-row output matrix of a
    /// package would take 347 MB of RAM at once).
    fn f32_weight(&self, t: &QTensor, rows: usize, cols: usize) -> Result<Tensor> {
        let step = self.chunk_rows(cols);
        if step >= rows {
            return self.f32_rows(t, 0, rows);
        }
        let out = Tensor::zeros((rows, cols), DType::F32, &self.dev)?;
        let mut r0 = 0;
        while r0 < rows {
            let n = step.min(rows - r0);
            out.slice_set(&self.f32_rows(t, r0, n)?, 0, r0)?;
            r0 += n;
        }
        Ok(out)
    }

    fn codec_quantizer(&self, format: QuantFormat, seg: usize) -> Result<Arc<TurboQuantizer>> {
        let key = (format.mode_id(), seg);
        let mut m = self.codecq.lock().unwrap();
        if let Some(q) = m.get(&key) {
            return Ok(Arc::clone(q));
        }
        // no rotation: the decoded rows are the weights themselves
        let q = Arc::new(TurboQuantizer::new(format, seg, RotationKind::None, 0, &self.dev)?);
        m.insert(key, Arc::clone(&q));
        Ok(q)
    }

    /// Codec rows: each row of the matrix cut into segments of `codec_segment(cols)` values, one
    /// codec row each (its scales inside).  Converted by row chunks on the device.
    fn coded_weight(&self, t: &QTensor, rows: usize, cols: usize, format: QuantFormat) -> Result<Weight> {
        let seg = codec_segment(cols, format).ok_or_else(|| msg(format!("{format}: no segment length for rows of {cols}")))?;
        let q = self.codec_quantizer(format, seg)?;
        let per = cols / seg;
        let step = self.chunk_rows(cols);
        let data = Tensor::zeros((rows * per, q.row_words()), DType::U32, &self.dev)?;
        let mut r0 = 0;
        while r0 < rows {
            let n = step.min(rows - r0);
            let x = self.f32_rows(t, r0, n)?;
            data.slice_set(&q.quantize_rows(&x.reshape((n * per, seg))?)?, 0, r0 * per)?;
            r0 += n;
        }
        Ok(Weight::Coded { rows, cols, data, q })
    }

    /// Store GGML blocks (from a GGUF file) as weight `name`, multiplied without expanding them.
    pub fn set_ggml(&mut self, name: &str, dtype: u32, rows: usize, cols: usize, data: &[u8]) -> Result<()> {
        self.mats.remove(name);
        let (mm, bytes) = crate::ggml::upload(crate::ggml::dtype_of(dtype)?, rows, cols, data, &self.dev)?;
        self.mats.insert(name.to_string(), Weight::Ggml { rows, cols, mm, bytes });
        Ok(())
    }

    /// Float32 rows `r0..r0 + n` of a half or FP8 matrix.
    fn decode_rows(&self, w: &Weight, r0: usize, n: usize) -> Result<Tensor> {
        match w {
            Weight::Dense(t) => t.narrow(0, r0, n),
            Weight::Ggml { .. } => self.dense(w)?.narrow(0, r0, n),
            Weight::Half(t) => t.narrow(0, r0, n)?.to_dtype(DType::F32),
            Weight::Coded { cols, data, q, .. } => {
                let per = cols / q.dim();
                q.dequantize_rows(&data.narrow(0, r0 * per, n * per)?)?.reshape((n, *cols))
            }
            Weight::Packed { .. } => self.dense(w)?.narrow(0, r0, n),
        }
    }

    fn vec(&self, name: &str) -> Result<&Tensor> {
        self.vecs.get(name).ok_or_else(|| msg(format!("missing tensor {name}")))
    }

    fn mat(&self, name: &str) -> Result<&Weight> {
        self.mats.get(name).ok_or_else(|| msg(format!("missing matrix {name}")))
    }

    #[cfg(feature = "rocm")]
    fn qw(parts: &[Option<Tensor>; 5]) -> Result<crate::hipk::QW> {
        let p = |i: usize| -> Result<*const std::ffi::c_void> {
            Ok(match &parts[i] {
                Some(t) => crate::hipk::ptr(t)? as *const _,
                None => std::ptr::null(),
            })
        };
        Ok(crate::hipk::QW { q: p(0)?, d: p(1)?, s: p(2)?, m: p(3)?, dm: p(4)? })
    }

    /// Float32 `[rows, cols]` of a weight (a scratch decode for packed weights).
    pub fn dense(&self, w: &Weight) -> Result<Tensor> {
        match w {
            Weight::Dense(t) => Ok(t.clone()),
            Weight::Half(_) | Weight::Coded { .. } => self.decode_rows(w, 0, w.rows()),
            Weight::Ggml { mm, .. } => match mm {
                QMatMul::QTensor(q) => q.dequantize(&self.dev),
                QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.to_dtype(DType::F32),
            },
            #[cfg(feature = "rocm")]
            Weight::Packed { scheme, rows, cols, parts } => {
                let out = unsafe { Tensor::empty((*rows, *cols), DType::F32, &self.dev) }?;
                self.hip.as_ref().unwrap().dequant(*scheme, Self::qw(parts)?, &out, rows * cols)?;
                Ok(out)
            }
            #[cfg(not(feature = "rocm"))]
            Weight::Packed { .. } => bail!("packed weights need the rocm build"),
        }
    }

    /// `x [n, cols] · Wᵀ + bias`.
    pub fn linear(&self, x: &Tensor, name: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.mat(name)?;
        let b = match bias {
            Some(b) => Some(self.vec(b)?),
            None => None,
        };
        self.linear_w(x, w, b)
    }

    pub fn linear_w(&self, x: &Tensor, w: &Weight, b: Option<&Tensor>) -> Result<Tensor> {
        let x = x.contiguous()?;
        let n = x.dims()[0];
        #[cfg(feature = "rocm")]
        if let Weight::Packed { scheme, rows, cols, parts } = w {
            if cols % 256 == 0 && (n <= self.gemv_max || n <= self.qgemm_max) {
                let y = unsafe { Tensor::empty((n, *rows), DType::F32, &self.dev) }?;
                let k = self.hip.as_ref().unwrap();
                if n <= self.gemv_max {
                    k.gemv(*scheme, Self::qw(parts)?, &x, b, &y, *rows, *cols, n, false)?;
                } else {
                    k.qgemm(*scheme, Self::qw(parts)?, &x, b, &y, *rows, *cols, n, false)?;
                }
                return Ok(y);
            }
        }
        let y = match w {
            // GGML blocks as they are: MMVQ (≤ 8 rows) / MMQ kernels, x quantised to Q8_1 on the fly
            Weight::Ggml { mm, .. } => mm.forward(&x)?,
            // native half matmul: x rounded to the weight's type, f32 accumulation in the BLAS
            Weight::Half(t) if self.half_compute == 1 => x.to_dtype(t.dtype())?.matmul(&t.t()?)?.to_dtype(DType::F32)?,
            // widen / decode row chunks of at most `decode_bytes`: the whole float32 matrix never
            // exists at once (the 59,283-row output matrix would take 347 MB)
            Weight::Half(_) | Weight::Coded { .. } => {
                let rows = w.rows();
                let step = self.chunk_rows(w.cols());
                if step >= rows {
                    x.matmul(&self.decode_rows(w, 0, rows)?.t()?)?
                } else {
                    let y = Tensor::zeros((n, rows), DType::F32, &self.dev)?;
                    let mut r0 = 0;
                    while r0 < rows {
                        let m = step.min(rows - r0);
                        y.slice_set(&x.matmul(&self.decode_rows(w, r0, m)?.t()?)?, 1, r0)?;
                        r0 += m;
                    }
                    y
                }
            }
            _ => x.matmul(&self.dense(w)?.t()?)?,
        };
        match b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn rms(&self, x: &Tensor, w: &str) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, self.vec(w)?, EPS)
    }

    /// `x + rms_norm(y)·w` – in place with one kernel on the custom path.
    fn add_rms(&self, x: Tensor, y: &Tensor, w: &str) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if self.custom && x.is_contiguous() {
            let y = y.contiguous()?;
            let (n, d) = x.dims2()?;
            self.hip.as_ref().unwrap().rms_norm_add(&x, &y, self.vec(w)?, EPS, n, d)?;
            return Ok(x);
        }
        x + self.rms(y, w)?
    }

    fn silu_mul(&self, g: &Tensor, u: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if self.custom {
            let out = unsafe { Tensor::empty(g.shape(), DType::F32, &self.dev) }?;
            self.hip.as_ref().unwrap().silu_mul(&g.contiguous()?, &u.contiguous()?, &out, g.elem_count())?;
            return Ok(out);
        }
        g.silu()? * u
    }

    fn swiglu(&self, u: &Tensor, inter: usize) -> Result<Tensor> {
        let n = u.dims()[0];
        #[cfg(feature = "rocm")]
        if self.custom {
            let out = unsafe { Tensor::empty((n, inter), DType::F32, &self.dev) }?;
            self.hip.as_ref().unwrap().swiglu_split(&u.contiguous()?, &out, n, inter)?;
            return Ok(out);
        }
        let _ = n;
        u.narrow(1, inter, inter)? * u.narrow(1, 0, inter)?.silu()?
    }

    /// Language attention of the pass's `t` rows (`q (1, t, 16, 128)`) over the cache rows
    /// `0..past + t`: on ROCm the custom kernels for the few rows of a decoding pass (`lm_attn`:
    /// 0 auto, 2 one block per row and head, 3 tiled), candle's flash attention for a prefill
    /// and elsewhere (or with `lm_attn` 1).
    #[allow(clippy::too_many_arguments)]
    fn lm_attend(&self, q: &Tensor, kc: &Tensor, vc: &Tensor, t: usize, past: usize, scale: f32, causal: bool,
                 quant: Option<&(TurboQuantizer, TurboQuantizer)>) -> Result<Tensor> {
        if let Some((kq, vq)) = quant {
            // compressed cache: attention on the packed rows (fused TurboQuant / FP8 kernels when
            // compiled in, else decoded + flash attention), q in (batch, heads, rows, dim)
            let qb = q.transpose(1, 2)?.contiguous()?;
            let mut cfg = FlashAttnConfig::new(scale);
            if causal {
                cfg = cfg.causal();
            }
            let o = candle_nn::turbo_quant::attention_packed(&qb, kc, vc, past + t, kq, vq, &cfg)?;
            return o.transpose(1, 2)?.contiguous()?.reshape((t, LM_HEADS * LM_DIM));
        }
        #[cfg(feature = "rocm")]
        if self.custom && self.lm_attn != 1 && t <= LM_ATTN_ROWS {
            let tiled = match self.lm_attn {
                2 => false,
                3 => true,
                _ => t >= LM_ATTN_TILED_FROM,
            };
            let out = unsafe { Tensor::empty((t, LM_HEADS * LM_DIM), DType::F32, &self.dev) }?;
            self.hip.as_ref().unwrap().lm_attention(&q.contiguous()?, kc, vc, &out, t, past, causal, scale, tiled)?;
            return Ok(out);
        }
        let keys = kc.narrow(0, 0, past + t)?.unsqueeze(0)?;
        let vals = vc.narrow(0, 0, past + t)?.unsqueeze(0)?;
        self.attend(q, &keys, &vals, scale, causal)?.reshape((t, LM_HEADS * LM_DIM))
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        // q (1, sq, hq, d), k / v (1, skv, hkv, d)
        if self.flash {
            return candle_nn::flash_attn::flash_attn(q, k, v, scale, causal);
        }
        let mut cfg = candle_nn::flash_attn::FlashAttnConfig::new(scale);
        if causal {
            cfg = cfg.causal();
        }
        cfg.fused = Some(false);
        candle_nn::flash_attn::flash_attn_with_config(q, k, v, &cfg)
    }

    /// Full (non-causal) attention in chunks of query rows: scores of a chunk by one batched
    /// matmul, the fused row softmax, then a matmul with the values.  The scores of a chunk
    /// never exceed `chunk_bytes`, whatever the image size.  q / k / v `(1, n, h, d)`.
    fn attend_chunked(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
        let (_, n, h, d) = q.dims4()?;
        let qh = q.squeeze(0)?.transpose(0, 1)?.contiguous()?; // (h, n, d)
        let kt = k.squeeze(0)?.transpose(0, 1)?.transpose(1, 2)?.contiguous()?; // (h, d, n)
        let vh = v.squeeze(0)?.transpose(0, 1)?.contiguous()?; // (h, n, d)
        let rows = (self.chunk_bytes / (h * n * 4)).clamp(16, n.max(16));
        let mut outs = Vec::with_capacity(n.div_ceil(rows));
        let mut c0 = 0;
        while c0 < n {
            let len = rows.min(n - c0);
            let qc = qh.narrow(1, c0, len)?.contiguous()?;
            let s = (qc.matmul(&kt)? * scale as f64)?;
            let p = candle_nn::ops::softmax_last_dim(&s)?;
            outs.push(p.matmul(&vh)?); // (h, len, d)
            c0 += len;
        }
        let o = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 1)? };
        o.transpose(0, 1)?.contiguous()?.reshape((1, n, h, d))
    }

    // --------------------------------------------------------------------------- vision
    /// `qkv [n, 3072]` → q, k, v `(1, n, 16, 64)`; q / k normalised and rotated (halves).
    fn vis_qkv(&self, i: usize, qkv: &Tensor, cos: &Tensor, sin: &Tensor, cos32: &Tensor, sin32: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let n = qkv.dims()[0];
        let p = format!("{VIS}blocks.{i}.");
        let qn = self.vec(&format!("{p}attn.q_norm.weight"))?;
        let kn = self.vec(&format!("{p}attn.k_norm.weight"))?;
        let shape = (1, n, VIS_HEADS, VIS_DIM);
        #[cfg(feature = "rocm")]
        if self.custom {
            let q = unsafe { Tensor::empty(shape, DType::F32, &self.dev) }?;
            let k = unsafe { Tensor::empty(shape, DType::F32, &self.dev) }?;
            let v = unsafe { Tensor::empty(shape, DType::F32, &self.dev) }?;
            self.hip.as_ref().unwrap().vis_qkv(&qkv.contiguous()?, qn, kn, cos, sin, EPS, &q, &k, &v, n)?;
            return Ok((q, k, v));
        }
        let _ = (cos, sin);
        let x = qkv.reshape((n, 3, VIS_HEADS, VIS_DIM))?;
        let part = |j: usize, w: Option<&Tensor>| -> Result<Tensor> {
            let t = x.narrow(1, j, 1)?.squeeze(1)?.contiguous()?; // (n, 16, 64)
            match w {
                None => t.unsqueeze(0),
                Some(w) => {
                    let t = candle_nn::ops::rms_norm(&t, w, EPS)?;
                    let t = t.transpose(0, 1)?.contiguous()?.unsqueeze(0)?; // (1, 16, n, 64)
                    let t = candle_nn::rotary_emb::rope(&t, cos32, sin32)?;
                    t.squeeze(0)?.transpose(0, 1)?.contiguous()?.unsqueeze(0)
                }
            }
        };
        Ok((part(0, Some(qn))?, part(1, Some(kn))?, part(2, None)?))
    }

    /// Full attention of every image of a batch on its own rows (q / k / v `(1, n, h, d)`).
    fn vis_attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, segs: &[usize]) -> Result<Tensor> {
        let scale = (VIS_DIM as f32).powf(-0.5);
        let one = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
            if self.vis_attn == 1 {
                self.attend_chunked(q, k, v, scale)
            } else {
                self.attend(q, k, v, scale, false)
            }
        };
        if segs.len() <= 1 {
            return one(q, k, v);
        }
        let mut outs = Vec::with_capacity(segs.len());
        let mut s0 = 0;
        for &len in segs {
            outs.push(one(&q.narrow(1, s0, len)?, &k.narrow(1, s0, len)?, &v.narrow(1, s0, len)?)?);
            s0 += len;
        }
        Tensor::cat(&outs, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn vis_block(&self, i: usize, x: Tensor, cos: &Tensor, sin: &Tensor, cos32: &Tensor, sin32: &Tensor, segs: &[usize]) -> Result<Tensor> {
        let n = x.dims()[0];
        let p = format!("{VIS}blocks.{i}.");
        let h = self.rms(&x, &format!("{p}norm1.weight"))?;
        let qkv = self.linear(&h, &format!("{p}attn.qkv.weight"), Some(&format!("{p}attn.qkv.bias")))?;
        let (q, k, v) = self.vis_qkv(i, &qkv, cos, sin, cos32, sin32)?;
        drop(qkv);
        let a = self.vis_attend(&q, &k, &v, segs)?;
        let a = a.reshape((n, VIS_WIDTH))?;
        let x = (x + self.linear(&a, &format!("{p}attn.proj.weight"), Some(&format!("{p}attn.proj.bias")))?)?;
        let h = self.rms(&x, &format!("{p}norm2.weight"))?;
        let g = self.linear(&h, &format!("{p}mlp.gate_proj.weight"), Some(&format!("{p}mlp.gate_proj.bias")))?;
        let u = self.linear(&h, &format!("{p}mlp.up_proj.weight"), Some(&format!("{p}mlp.up_proj.bias")))?;
        let m = self.silu_mul(&g, &u)?;
        x + self.linear(&m, &format!("{p}mlp.down_proj.weight"), Some(&format!("{p}mlp.down_proj.bias")))?
    }

    /// Patches `[n, 588]` + rotary tables `[n, 64]` → image tokens `[n/4, 1536]`.  `segs` are the
    /// patch counts of the images of a batch (each a multiple of 4, summing to `n`): rows are
    /// processed together, attention stays inside each image.
    pub fn vision(&self, patches: &[f32], n: usize, cos: &[f32], sin: &[f32], segs: &[usize]) -> Result<Vec<f32>> {
        if n % 4 != 0 || patches.len() != n * PATCH_ROW || cos.len() != n * 64 || sin.len() != n * 64 {
            bail!("vision: bad input sizes");
        }
        if !segs.is_empty() && (segs.iter().sum::<usize>() != n || segs.iter().any(|s| s % 4 != 0 || *s == 0)) {
            bail!("vision: batch segments must be non-empty multiples of 4 summing to n");
        }
        let x = Tensor::from_slice(patches, (n, PATCH_ROW), &self.dev)?;
        let cos = Tensor::from_slice(cos, (n, 64), &self.dev)?;
        let sin = Tensor::from_slice(sin, (n, 64), &self.dev)?;
        let cos32 = cos.narrow(1, 0, 32)?.contiguous()?;
        let sin32 = sin.narrow(1, 0, 32)?.contiguous()?;
        let mut x = self.linear(&x, &format!("{VIS}patch_embed.proj.weight"), Some(&format!("{VIS}patch_embed.proj.bias")))?;
        for i in 0..VIS_DEPTH {
            x = self.vis_block(i, x, &cos, &sin, &cos32, &sin32, segs)?;
        }
        let x = self.rms(&x, &format!("{VIS}post_layernorm.weight"))?;
        let x = x.reshape((n / 4, 4 * VIS_WIDTH))?;
        let x = self.linear(&x, &format!("{VIS}downsample.weight"), Some(&format!("{VIS}downsample.bias")))?;
        let h = self.linear(&x, &format!("{VIS}merger.proj.weight"), None)?;
        let h = candle_nn::ops::layer_norm(
            &h,
            self.vec(&format!("{VIS}merger.post_projection_norm.weight"))?,
            self.vec(&format!("{VIS}merger.post_projection_norm.bias"))?,
            EPS,
        )?
        .gelu_erf()?;
        let g = self.linear(&h, &format!("{VIS}merger.gate_proj.weight"), None)?;
        let u = self.linear(&h, &format!("{VIS}merger.up_proj.weight"), None)?;
        let m = self.silu_mul(&g, &u)?;
        let out = self.linear(&m, &format!("{VIS}merger.down_proj.weight"), None)?;
        out.flatten_all()?.to_vec1::<f32>()
    }

    // --------------------------------------------------------------------------- language
    /// Quantizers (keys, values) of a KV cache in `format`, `None` for float32.  Keys and values
    /// get different seeds of the rotation (the TurboQuant reference: values `seed + 2`).
    pub fn kv_quantizers(&self, format: u32) -> Result<Option<Arc<(TurboQuantizer, TurboQuantizer)>>> {
        let Some(f) = kv_quant_format(format) else {
            return Ok(None);
        };
        let k = TurboQuantizer::new(f, LM_DIM, RotationKind::Auto, 0x6772_6176, &self.dev)?;
        let v = TurboQuantizer::new(f, LM_DIM, RotationKind::Auto, 0x6772_6176 + 2, &self.dev)?;
        Ok(Some(Arc::new((k, v))))
    }

    /// Bytes of one cached position (all layers, keys and values) in `format`.
    pub fn kv_row_bytes(&self, format: u32) -> Result<usize> {
        Ok(match self.kv_quantizers(format)? {
            None => 2 * LM_DEPTH * LM_KV * LM_DIM * 4,
            Some(q) => LM_DEPTH * LM_KV * (q.0.row_words() + q.1.row_words()) * 4,
        })
    }

    fn cache_tensor(&self, cap: usize, words: Option<usize>) -> Result<Tensor> {
        match words {
            None => unsafe { Tensor::empty((cap, LM_KV, LM_DIM), DType::F32, &self.dev) },
            Some(w) => Tensor::zeros((1, LM_KV, cap, w), DType::U32, &self.dev),
        }
    }

    pub fn new_cache(&self, cap: usize) -> Result<Cache> {
        let quant = self.kv_quantizers(self.kv_format)?;
        let (kw, vw) = match &quant {
            Some(q) => (Some(q.0.row_words()), Some(q.1.row_words())),
            None => (None, None),
        };
        let mut k = Vec::with_capacity(LM_DEPTH);
        let mut v = Vec::with_capacity(LM_DEPTH);
        for _ in 0..LM_DEPTH {
            k.push(self.cache_tensor(cap, kw)?);
            v.push(self.cache_tensor(cap, vw)?);
        }
        Ok(Cache { k, v, cap, quant })
    }

    /// Grow a cache to `cap` rows, keeping the first `len`.
    pub fn grow_cache(&self, c: &mut Cache, cap: usize, len: usize) -> Result<()> {
        if cap <= c.cap {
            return Ok(());
        }
        let words = c.quant.as_ref().map(|q| (q.0.row_words(), q.1.row_words()));
        for l in 0..LM_DEPTH {
            for (t, w) in [(&mut c.k[l], words.map(|w| w.0)), (&mut c.v[l], words.map(|w| w.1))] {
                let new = self.cache_tensor(cap, w)?;
                if len > 0 {
                    match w {
                        None => new.slice_set(&t.narrow(0, 0, len)?.contiguous()?, 0, 0)?,
                        Some(_) => new.slice_set(&t.narrow(2, 0, len)?.contiguous()?, 2, 0)?,
                    }
                }
                *t = new;
            }
        }
        c.cap = cap;
        Ok(())
    }

    /// Rotate q / k (adjacent pairs) and write k / v into the cache rows `past..past+t`;
    /// returns q `(1, t, 16, 128)`.
    #[allow(clippy::too_many_arguments)]
    fn rope_cache(&self, q: Tensor, k: Tensor, v: Tensor, cos: &Tensor, sin: &Tensor, kc: &Tensor, vc: &Tensor, past: usize,
                  quant: Option<&(TurboQuantizer, TurboQuantizer)>) -> Result<Tensor> {
        let t = q.dims()[0];
        #[cfg(feature = "rocm")]
        if self.custom && quant.is_none() {
            let q = q.contiguous()?;
            self.hip.as_ref().unwrap().lm_rope_cache(&q, &k.contiguous()?, &v.contiguous()?, cos, sin, kc, vc, t, past)?;
            return q.reshape((1, t, LM_HEADS, LM_DIM));
        }
        let rot = |x: Tensor, heads: usize| -> Result<Tensor> {
            let x = x.reshape((1, t, heads, LM_DIM))?.transpose(1, 2)?.contiguous()?;
            let x = candle_nn::rotary_emb::rope_i(&x, cos, sin)?;
            x.transpose(1, 2)?.contiguous()
        };
        let q = rot(q, LM_HEADS)?;
        let k = rot(k, LM_KV)?.squeeze(0)?;
        let v = v.reshape((t, LM_KV, LM_DIM))?;
        match quant {
            None => {
                kc.slice_set(&k, 0, past)?;
                vc.slice_set(&v.contiguous()?, 0, past)?;
            }
            Some((kq, vq)) => {
                // (t, 8, 128) → head-major rows (8·t, 128) → packed (1, 8, t, words) at `past`
                let pack = |qz: &TurboQuantizer, x: &Tensor| -> Result<Tensor> {
                    let rows = x.transpose(0, 1)?.contiguous()?.reshape((LM_KV * t, LM_DIM))?;
                    qz.quantize_rows(&rows)?.reshape((1, LM_KV, t, qz.row_words()))
                };
                kc.slice_set(&pack(kq, &k)?, 2, past)?;
                vc.slice_set(&pack(vq, &v)?, 2, past)?;
            }
        }
        Ok(q)
    }

    /// `t` rows on top of `past` cached ones; returns the logits of rows `from..t` and, with
    /// `rest`, for each of those rows the best of the output rows a sliced head cut (`REST`)
    /// as `[logit, row]` pairs – the caller takes that token where it beats every kept row.
    #[allow(clippy::too_many_arguments)]
    pub fn lm(&self, c: &Cache, x: &[f32], t: usize, cos: &[f32], sin: &[f32], past: usize, causal: bool, from: usize, inject: Option<&[f32]>,
              rest: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        if past + t > c.cap {
            bail!("lm: cache too small ({} + {} > {})", past, t, c.cap);
        }
        if x.len() != t * LM_WIDTH || cos.len() != t * 64 || sin.len() != t * 64 || from >= t {
            bail!("lm: bad input sizes");
        }
        let mut lap = if profiling() {
            self.dev.synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let mut x = Tensor::from_slice(x, (t, LM_WIDTH), &self.dev)?;
        let cos = Tensor::from_slice(cos, (t, 64), &self.dev)?;
        let sin = Tensor::from_slice(sin, (t, 64), &self.dev)?;
        let scale = (LM_DIM as f32).powf(-0.5);
        self.lap("upload", &mut lap)?;
        for i in 0..LM_DEPTH {
            let p = format!("{LM}layers.{i}.");
            let h = self.rms(&x, &format!("{p}input_layernorm.weight"))?;
            self.lap("norm", &mut lap)?;
            let q = self.linear(&h, &format!("{p}self_attn.q_proj.weight"), None)?;
            let k = self.linear(&h, &format!("{p}self_attn.k_proj.weight"), None)?;
            let v = self.linear(&h, &format!("{p}self_attn.v_proj.weight"), None)?;
            self.lap("qkv", &mut lap)?;
            let quant = c.quant.as_deref();
            let q = self.rope_cache(q, k, v, &cos, &sin, &c.k[i], &c.v[i], past, quant)?;
            self.lap("rope_cache", &mut lap)?;
            let o = self.lm_attend(&q, &c.k[i], &c.v[i], t, past, scale, causal, quant)?;
            self.lap("attention", &mut lap)?;
            let a = self.linear(&o, &format!("{p}self_attn.o_proj.weight"), None)?;
            self.lap("o", &mut lap)?;
            x = self.add_rms(x, &a, &format!("{p}post_self_attn_layernorm.weight"))?;
            let h = self.rms(&x, &format!("{p}post_attention_layernorm.weight"))?;
            self.lap("norm", &mut lap)?;
            let u = self.linear(&h, &format!("{p}mlp.gate_up_proj.weight"), None)?;
            self.lap("gate_up", &mut lap)?;
            let m = self.swiglu(&u, LM_INTER)?;
            self.lap("swiglu", &mut lap)?;
            let d = self.linear(&m, &format!("{p}mlp.down_proj.weight"), None)?;
            self.lap("down", &mut lap)?;
            x = self.add_rms(x, &d, &format!("{p}post_mlp_layernorm.weight"))?;
            if i == 0 {
                if let Some(r) = inject {
                    // UnifiedNgram residual rows (zero where no trigram matched)
                    x = (x + Tensor::from_slice(r, (t, LM_WIDTH), &self.dev)?)?;
                }
            }
            self.lap("norm", &mut lap)?;
        }
        let xs = x.narrow(0, from, t - from)?;
        let h = self.rms(&xs, &format!("{LM}norm.weight"))?;
        let logits = self.linear(&h, HEAD, None)?;
        self.lap("head", &mut lap)?;
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        self.lap("logits_to_host", &mut lap)?;
        let mut best = Vec::new();
        if rest {
            let r = self.linear(&h, REST, None)?;
            let vals = r.max_keepdim(1)?.flatten_all()?.to_vec1::<f32>()?;
            let rows = r.argmax_keepdim(1)?.flatten_all()?.to_vec1::<u32>()?;
            best = vals.into_iter().zip(rows).flat_map(|(v, i)| [v, i as f32]).collect();
            self.lap("cut_rows", &mut lap)?;
        }
        Ok((logits, best))
    }

    pub fn head_rows(&self) -> usize {
        self.mats.get(HEAD).map(|w| w.rows()).unwrap_or(0)
    }

    /// Evaluate the cut output rows (`lm_head.rest.weight`) on a batch of uncertain hidden states
    /// `x [n_rows, k_dim]` (passed after Cauchy-Schwarz gating) and return `(max_logits, argmax_row_ids)`.
    pub fn eval_rest_max(&self, x: &[f32], n_rows: usize, k_dim: usize) -> Result<(Vec<f32>, Vec<u32>)> {
        if n_rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let h = Tensor::from_slice(x, (n_rows, k_dim), &self.dev)?;
        let r = self.linear(&h, REST, None)?;
        let vals = r.max_keepdim(1)?.flatten_all()?.to_vec1::<f32>()?;
        let rows = r.argmax_keepdim(1)?.flatten_all()?.to_vec1::<u32>()?;
        Ok((vals, rows))
    }

    // --------------------------------------------------------------------------- self-check entry points
    /// An engine on the same device and kernels without weights.
    /// Profile step: with `GRAVITY_GPU_PROFILE=1`, wait for the device and book the time since
    /// the last step under `what`.
    fn lap(&self, what: &'static str, t: &mut Option<Instant>) -> Result<()> {
        if let Some(t0) = t {
            self.dev.synchronize()?;
            let mut m = TIMES.lock().unwrap();
            let e = m.entry(what).or_insert((0.0, 0));
            e.0 += t0.elapsed().as_secs_f64();
            e.1 += 1;
            *t = Some(Instant::now());
        }
        Ok(())
    }

    fn scratch(&self) -> Engine {
        Engine {
            dev: self.dev.clone(),
            kind: self.kind,
            #[cfg(feature = "rocm")]
            hip: self.hip.clone(),
            mats: HashMap::new(),
            vecs: HashMap::new(),
            gemv_max: self.gemv_max,
            qgemm_max: self.qgemm_max,
            custom: self.custom,
            vis_attn: self.vis_attn,
            lm_attn: self.lm_attn,
            chunk_bytes: self.chunk_bytes,
            flash: self.flash,
            weight_store: self.weight_store,
            half_compute: self.half_compute,
            decode_bytes: self.decode_bytes,
            kv_format: self.kv_format,
            codecq: Mutex::new(HashMap::new()),
            note: String::new(),
        }
    }

    pub fn test_linear(&self, t: &QTensor, x: &[f32], n: usize, bias: Option<&[f32]>) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        tmp.set("w", t, false)?;
        let xt = Tensor::from_slice(x, (n, t.cols as usize), &self.dev)?;
        let b = match bias {
            Some(b) => Some(Tensor::from_slice(b, b.len(), &self.dev)?),
            None => None,
        };
        tmp.linear_w(&xt, tmp.mat("w")?, b.as_ref())?.flatten_all()?.to_vec1::<f32>()
    }

    /// Seconds per `x [n, cols] · Wᵀ` with path 0 auto, 1 gemv, 2 tiled fused gemm, 3 decode + BLAS.
    pub fn bench_linear(&self, t: &QTensor, n: usize, path: u32, reps: usize) -> Result<f64> {
        let mut tmp = self.scratch();
        match path {
            1 => (tmp.gemv_max, tmp.qgemm_max) = (usize::MAX, 0),
            2 => (tmp.gemv_max, tmp.qgemm_max) = (0, usize::MAX),
            3 => (tmp.gemv_max, tmp.qgemm_max) = (0, 0),
            _ => {}
        }
        if path == 1 && n > 32 {
            bail!("gemv takes at most 32 rows");
        }
        tmp.set("w", t, false)?;
        let x = Tensor::ones((n, t.cols as usize), DType::F32, &self.dev)?;
        let w = tmp.mat("w")?;
        tmp.linear_w(&x, w, None)?;
        self.dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            tmp.linear_w(&x, w, None)?;
        }
        self.dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    /// `x [n, cols] · Wᵀ` with `W` given as GGML blocks (the device's quantised matmul).
    pub fn test_ggml_linear(&self, dtype: u32, rows: usize, cols: usize, data: &[u8], x: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        tmp.set_ggml("w", dtype, rows, cols, data)?;
        let xt = Tensor::from_slice(x, (n, cols), &self.dev)?;
        tmp.linear_w(&xt, tmp.mat("w")?, None)?.flatten_all()?.to_vec1::<f32>()
    }

    pub fn test_dequant(&self, t: &QTensor) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        tmp.set("w", t, false)?;
        tmp.dense(tmp.mat("w")?)?.flatten_all()?.to_vec1::<f32>()
    }

    /// One attention layer of each tower on given inputs (weights for the norms in `vecs`).
    #[allow(clippy::too_many_arguments)]
    pub fn test_lm_attention(&self, q: &[f32], k: &[f32], v: &[f32], cos: &[f32], sin: &[f32], kold: &[f32], vold: &[f32], past: usize, t: usize, causal: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        let cap = past + t;
        let kc = Tensor::zeros((cap, LM_KV, LM_DIM), DType::F32, &self.dev)?;
        let vc = Tensor::zeros((cap, LM_KV, LM_DIM), DType::F32, &self.dev)?;
        if past > 0 {
            kc.slice_set(&Tensor::from_slice(kold, (past, LM_KV, LM_DIM), &self.dev)?, 0, 0)?;
            vc.slice_set(&Tensor::from_slice(vold, (past, LM_KV, LM_DIM), &self.dev)?, 0, 0)?;
        }
        let q = Tensor::from_slice(q, (t, LM_HEADS * LM_DIM), &self.dev)?;
        let k = Tensor::from_slice(k, (t, LM_KV * LM_DIM), &self.dev)?;
        let v = Tensor::from_slice(v, (t, LM_KV * LM_DIM), &self.dev)?;
        let cos = Tensor::from_slice(cos, (t, 64), &self.dev)?;
        let sin = Tensor::from_slice(sin, (t, 64), &self.dev)?;
        let q = self.rope_cache(q, k, v, &cos, &sin, &kc, &vc, past, None)?;
        let o = self.lm_attend(&q, &kc, &vc, t, past, (LM_DIM as f32).powf(-0.5), causal, None)?;
        Ok((o.flatten_all()?.to_vec1()?, kc.flatten_all()?.to_vec1()?))
    }

    /// The attention output of [`Self::test_lm_attention`] through a cache in `format` (the
    /// rows `kold` / `vold` are written through the same cache path first).
    #[allow(clippy::too_many_arguments)]
    pub fn test_kv_attention(&self, format: u32, q: &[f32], k: &[f32], v: &[f32], cos: &[f32], sin: &[f32], kold: &[f32], vold: &[f32],
                             past: usize, t: usize, causal: bool) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        tmp.kv_format = format;
        let c = tmp.new_cache(past + t)?;
        let quant = c.quant.as_deref();
        let ones = Tensor::ones((past.max(1), 64), DType::F32, &self.dev)?;
        let zeros = Tensor::zeros((past.max(1), 64), DType::F32, &self.dev)?;
        if past > 0 {
            // the old rows are stored as they are (cos 1, sin 0: no rotation)
            let qo = Tensor::zeros((past, LM_HEADS * LM_DIM), DType::F32, &self.dev)?;
            let ko = Tensor::from_slice(kold, (past, LM_KV * LM_DIM), &self.dev)?;
            let vo = Tensor::from_slice(vold, (past, LM_KV * LM_DIM), &self.dev)?;
            tmp.rope_cache(qo, ko, vo, &ones, &zeros, &c.k[0], &c.v[0], 0, quant)?;
        }
        let q = Tensor::from_slice(q, (t, LM_HEADS * LM_DIM), &self.dev)?;
        let k = Tensor::from_slice(k, (t, LM_KV * LM_DIM), &self.dev)?;
        let v = Tensor::from_slice(v, (t, LM_KV * LM_DIM), &self.dev)?;
        let cos = Tensor::from_slice(cos, (t, 64), &self.dev)?;
        let sin = Tensor::from_slice(sin, (t, 64), &self.dev)?;
        let q = tmp.rope_cache(q, k, v, &cos, &sin, &c.k[0], &c.v[0], past, quant)?;
        let o = tmp.lm_attend(&q, &c.k[0], &c.v[0], t, past, (LM_DIM as f32).powf(-0.5), causal, quant)?;
        o.flatten_all()?.to_vec1()
    }

    pub fn test_vis_attention(&self, qkv: &[f32], qn: &[f32], kn: &[f32], cos: &[f32], sin: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        tmp.vecs.insert(format!("{VIS}blocks.0.attn.q_norm.weight"), Tensor::from_slice(qn, VIS_DIM, &self.dev)?);
        tmp.vecs.insert(format!("{VIS}blocks.0.attn.k_norm.weight"), Tensor::from_slice(kn, VIS_DIM, &self.dev)?);
        let qkv = Tensor::from_slice(qkv, (n, 3 * VIS_WIDTH), &self.dev)?;
        let cos = Tensor::from_slice(cos, (n, 64), &self.dev)?;
        let sin = Tensor::from_slice(sin, (n, 64), &self.dev)?;
        let cos32 = cos.narrow(1, 0, 32)?.contiguous()?;
        let sin32 = sin.narrow(1, 0, 32)?.contiguous()?;
        let (q, k, v) = tmp.vis_qkv(0, &qkv, &cos, &sin, &cos32, &sin32)?;
        let a = tmp.vis_attend(&q, &k, &v, &[])?;
        a.flatten_all()?.to_vec1()
    }

    /// 0: rms_norm(x, w), 1: layer_norm(x, w, b), 2: swiglu (x is [n, 2d]), 3: silu(x) · w-rows (x, y [n, d])
    pub fn test_rows(&self, op: u32, x: &[f32], n: usize, d: usize, w: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let mut tmp = self.scratch();
        let out = match op {
            0 => {
                tmp.vecs.insert("w".into(), Tensor::from_slice(w, d, &self.dev)?);
                tmp.rms(&Tensor::from_slice(x, (n, d), &self.dev)?, "w")?
            }
            1 => candle_nn::ops::layer_norm(&Tensor::from_slice(x, (n, d), &self.dev)?, &Tensor::from_slice(w, d, &self.dev)?, &Tensor::from_slice(b, d, &self.dev)?, EPS)?,
            2 => tmp.swiglu(&Tensor::from_slice(x, (n, 2 * d), &self.dev)?, d)?,
            3 => tmp.silu_mul(&Tensor::from_slice(x, (n, d), &self.dev)?, &Tensor::from_slice(w, (n, d), &self.dev)?)?,
            _ => bail!("unknown op {op}"),
        };
        out.flatten_all()?.to_vec1()
    }
}
