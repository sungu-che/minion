//! GGML block formats (the k-quants of llama.cpp, through candle) – the "pyramid" GGUF path.
//!
//! * `quantize`: a source tensor (checkpoint bf16, a reshaped f32 array, or a gravity package
//!   part) into GGML blocks on the CPU, by row chunks in parallel – what `tools/build_gguf.py`
//!   writes into the GGUF file;
//! * `dequant`: GGML blocks back to float32 on the CPU (references, host-side reads);
//! * `upload`: GGML blocks onto the device as a candle `QMatMul`, which the engine multiplies
//!   without expanding them (CUDA: MMVQ for up to 8 rows, MMQ above, activations as Q8_1);
//! * `gguf_info`: candle's own GGUF reader on a file, to check that a written file is standard.

use candle_core::quantized::{ggml_file, gguf_file, GgmlDType, QMatMul, QTensor as CandleQ};
use candle_core::{bail, Device, Result, Tensor};
use gravity_kernels::codec::{self, QTensor};
use std::sync::Mutex;

/// GGML type id (`ggml.h`) → candle's dtype.
pub fn dtype_of(id: u32) -> Result<GgmlDType> {
    Ok(match id {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        30 => GgmlDType::BF16,
        _ => bail!("unsupported GGML type {id}"),
    })
}

/// Bytes of one row of `cols` values; the row must hold whole blocks.
pub fn row_bytes(dt: GgmlDType, cols: usize) -> Result<usize> {
    let bs = dt.block_size();
    if cols == 0 || cols % bs != 0 {
        bail!("{dt:?}: a row of {cols} values is not a whole number of {bs}-value blocks");
    }
    Ok(cols / bs * dt.type_size())
}

/// Float32 values of a source tensor go through at most this many bytes per task.
const CHUNK_BYTES: usize = 16 << 20;

/// Quantise `t` (rows × cols, any gravity codec scheme) into `out` as GGML `dt` blocks, row
/// chunks in parallel.  `imatrix` (one importance weight per column) selects llama.cpp's
/// importance-weighted search where candle has it.  Returns the bytes written.
///
/// # Safety
/// `t`'s pointers must be valid for its shape.
pub unsafe fn quantize(t: &QTensor, dt: GgmlDType, imatrix: Option<&[f32]>, out: &mut [u8]) -> Result<usize> {
    t.check().map_err(|c| candle_core::Error::Msg(format!("tensor parts do not match the shape (code {c})")))?;
    let (rows, cols) = (t.rows as usize, t.cols as usize);
    let rb = row_bytes(dt, cols)?;
    let need = rows * rb;
    if out.len() < need {
        bail!("quantize: {need} bytes needed, {} given", out.len());
    }
    if let Some(w) = imatrix {
        if w.len() != cols {
            bail!("quantize: {} importance weights for rows of {cols}", w.len());
        }
    }
    let step = (CHUNK_BYTES / (cols * 4)).max(1);
    let dst = gravity_kernels::par::Ptr(out.as_mut_ptr());
    let err: Mutex<Option<String>> = Mutex::new(None);
    let tc = *t;
    gravity_kernels::par::chunks(rows, step, |a, b| {
        let n = b - a;
        let run = || -> Result<Vec<u8>> {
            let mut v = vec![0f32; n * cols];
            codec::dequant_range(&tc, a * cols, n * cols, v.as_mut_ptr());
            let x = Tensor::from_vec(v, (n, cols), &Device::Cpu)?;
            let q = match imatrix {
                Some(w) => CandleQ::quantize_imatrix(&x, w, dt)?,
                None => CandleQ::quantize(&x, dt)?,
            };
            Ok(q.data()?.into_owned())
        };
        match run() {
            Ok(bytes) if bytes.len() == n * rb => std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.get().add(a * rb), bytes.len()),
            Ok(bytes) => *err.lock().unwrap() = Some(format!("quantize: rows {a}..{b} gave {} bytes, {} expected", bytes.len(), n * rb)),
            Err(e) => *err.lock().unwrap() = Some(e.to_string()),
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        bail!("{e}");
    }
    Ok(need)
}

/// GGML blocks of a `rows × cols` tensor as float32 (CPU).
pub fn dequant(dt: GgmlDType, rows: usize, cols: usize, data: &[u8], out: &mut [f32]) -> Result<()> {
    if data.len() < rows * row_bytes(dt, cols)? || out.len() < rows * cols {
        bail!("dequant: buffers too small for {rows} × {cols} {dt:?}");
    }
    let q = ggml_file::qtensor_from_ggml(dt, &data[..rows * row_bytes(dt, cols)?], vec![rows, cols], &Device::Cpu)?;
    let v = q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    out[..v.len()].copy_from_slice(&v);
    Ok(())
}

/// GGML blocks onto `dev` as a matmul weight `[rows, cols]` (`y = x · Wᵀ`).
pub fn upload(dt: GgmlDType, rows: usize, cols: usize, data: &[u8], dev: &Device) -> Result<(QMatMul, usize)> {
    let len = rows * row_bytes(dt, cols)?;
    if data.len() < len {
        bail!("upload: {len} bytes needed for {rows} × {cols} {dt:?}, {} given", data.len());
    }
    let q = ggml_file::qtensor_from_ggml(dt, &data[..len], vec![rows, cols], dev)?;
    let bytes = q.storage_size_in_bytes();
    Ok((QMatMul::from_qtensor(q)?, bytes))
}

/// `(tensors, metadata entries)` of a GGUF file, read with candle's reader.
pub fn gguf_info(path: &str) -> Result<(usize, usize)> {
    let mut f = std::fs::File::open(path)?;
    let c = gguf_file::Content::read(&mut f)?;
    Ok((c.tensor_infos.len(), c.metadata.len()))
}
