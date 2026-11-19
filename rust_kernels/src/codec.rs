//! The weight codecs of `gravity_ocr/quant.py`, decoded bit for bit.
//!
//! A tensor is a flat vector cut into super-blocks of 256 values.  Every
//! packed stream (`q`, and the per-block scale indices `s` / `m`) is a
//! little-endian bit stream in which value `i` occupies bits
//! `[i·bits, (i+1)·bits)`; a super-block always starts on a byte boundary, so
//! super-block `sb` of the `q` stream starts at byte `sb·32·bits`.
//!
//! The arithmetic is the one `quant.dequantize` does in float32, one rounding
//! per numpy operation and never a fused multiply-add:
//!
//! * `q8`  `q · d[block]`
//! * `nf4` `LEVEL[q] · d[block]`
//! * symmetric `q6` `q3`  `(q − 2^(bits−1)) · (d[super] · (s / levels))`
//! * asymmetric `q5` `q4` `q · (d[super] · (s / levels)) − dm[super] · (m / levels)`

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub const SUPER: usize = 256;

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const BF16: u32 = 2;
pub const Q8: u32 = 3;
pub const Q6: u32 = 4;
pub const Q5: u32 = 5;
pub const Q4: u32 = 6;
pub const Q3: u32 = 7;
pub const NF4: u32 = 8;

/// `quant.NF4_LEVELS` as float32 bit patterns (dumped from numpy).
const NF4_BITS: [u32; 16] = [
    0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
    0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000,
];

#[inline]
fn nf4(i: usize) -> f32 {
    f32::from_bits(NF4_BITS[i])
}

/// One tensor as the Python side hands it over: pointers into the packed
/// parts (usually straight into the memory-mapped package) and their sizes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QTensor {
    pub scheme: u32,
    pub _pad: u32,
    pub rows: u64,
    pub cols: u64,
    /// `q` for quantised schemes, `w` for the dense ones.
    pub q: *const u8,
    pub q_len: u64,
    pub d: *const u8,
    pub d_len: u64,
    pub s: *const u8,
    pub s_len: u64,
    pub m: *const u8,
    pub m_len: u64,
    pub dm: *const u8,
    pub dm_len: u64,
}
unsafe impl Send for QTensor {}
unsafe impl Sync for QTensor {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub bits: u32,
    pub block: usize,
    pub symmetric: bool,
    pub scale_bits: u32,
}

pub fn spec(scheme: u32) -> Option<Spec> {
    let s = |bits, block, symmetric, scale_bits| Some(Spec { bits, block, symmetric, scale_bits });
    match scheme {
        F32 => s(32, 1, true, 0),
        F16 | BF16 => s(16, 1, true, 0),
        Q8 => s(8, 32, true, 0),
        Q6 => s(6, 16, true, 8),
        Q5 => s(5, 32, false, 6),
        Q4 => s(4, 32, false, 6),
        Q3 => s(3, 16, true, 6),
        NF4 => s(4, 64, true, 0),
        _ => None,
    }
}

pub fn is_dense(scheme: u32) -> bool {
    matches!(scheme, F32 | F16 | BF16)
}

pub const ERR_SCHEME: i32 = -1;
pub const ERR_NULL: i32 = -2;
pub const ERR_SHORT: i32 = -3;
pub const ERR_SHAPE: i32 = -4;

impl QTensor {
    pub fn numel(&self) -> usize {
        (self.rows * self.cols) as usize
    }

    pub fn supers(&self) -> usize {
        self.numel().div_ceil(SUPER)
    }

    /// Every pointer is there and every part is long enough for the shape.
    pub fn check(&self) -> Result<(), i32> {
        let sp = spec(self.scheme).ok_or(ERR_SCHEME)?;
        let n = self.numel();
        let need = |p: *const u8, len: u64, want: usize| -> Result<(), i32> {
            if want == 0 {
                return Ok(());
            }
            if p.is_null() {
                return Err(ERR_NULL);
            }
            if (len as usize) < want {
                return Err(ERR_SHORT);
            }
            Ok(())
        };
        if is_dense(self.scheme) {
            return need(self.q, self.q_len, n * (sp.bits as usize / 8));
        }
        let nsup = self.supers();
        let nblk = nsup * SUPER / sp.block;
        need(self.q, self.q_len, nsup * 32 * sp.bits as usize)?;
        match self.scheme {
            Q8 | NF4 => need(self.d, self.d_len, nblk * 2),
            _ => {
                let per_super = SUPER / sp.block;
                let sbytes = nsup * per_super * sp.scale_bits as usize / 8;
                need(self.d, self.d_len, nsup * 2)?;
                need(self.s, self.s_len, sbytes)?;
                if !sp.symmetric {
                    need(self.dm, self.dm_len, nsup * 2)?;
                    need(self.m, self.m_len, sbytes)?;
                }
                Ok(())
            }
        }
    }

    /// The fused matmul works on whole super-blocks of one row.
    pub fn rows_are_supers(&self) -> bool {
        is_dense(self.scheme) || self.cols as usize % SUPER == 0
    }
}

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
unsafe fn rd16(p: *const u8, i: usize) -> u16 {
    (p.add(2 * i) as *const u16).read_unaligned()
}

/// `bits` wide value `i` of a little-endian bit stream that has `len` bytes.
#[inline]
unsafe fn bits_at(p: *const u8, len: usize, i: usize, bits: u32) -> u32 {
    let pos = i * bits as usize;
    let byte = pos >> 3;
    let shift = (pos & 7) as u32;
    let mut v = *p.add(byte) as u32;
    if shift + bits > 8 && byte + 1 < len {
        v |= (*p.add(byte + 1) as u32) << 8;
    }
    (v >> shift) & ((1u32 << bits) - 1)
}

/// `idx / levels` as numpy computes it (float32 division of the index).
#[inline]
fn ratio(idx: u32, levels: u32) -> f32 {
    idx as f32 / levels as f32
}

/// Per-block multipliers of super-block `sb` of a k-quant style scheme.
#[inline]
unsafe fn scales(t: &QTensor, sp: &Spec, sb: usize, dq: &mut [f32; 16], mq: &mut [f32; 16]) {
    let per_super = SUPER / sp.block;
    let levels = (1u32 << sp.scale_bits) - 1;
    let d = f16_to_f32(rd16(t.d, sb));
    for b in 0..per_super {
        let i = sb * per_super + b;
        dq[b] = d * ratio(bits_at(t.s, t.s_len as usize, i, sp.scale_bits), levels);
    }
    if !sp.symmetric {
        let dm = f16_to_f32(rd16(t.dm, sb));
        for b in 0..per_super {
            let i = sb * per_super + b;
            mq[b] = dm * ratio(bits_at(t.m, t.m_len as usize, i, sp.scale_bits), levels);
        }
    }
}

/// Decode super-block `sb` of a quantised tensor (all 256 values, including
/// the zero padding of the last one).
pub unsafe fn decode_super(t: &QTensor, sb: usize, out: &mut [f32; SUPER]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::fast() {
        let sp = spec(t.scheme).unwrap();
        // the vector path loads 8 bytes per group of 8 values
        let end = (sb + 1) * 32 * sp.bits as usize;
        if end + 8 <= t.q_len as usize {
            decode_super_avx2(t, sb, out);
            return;
        }
    }
    decode_super_scalar(t, sb, out);
}

pub unsafe fn decode_super_scalar(t: &QTensor, sb: usize, out: &mut [f32; SUPER]) {
    let sp = spec(t.scheme).unwrap();
    let q = t.q.add(sb * 32 * sp.bits as usize);
    match t.scheme {
        Q8 => {
            for b in 0..SUPER / 32 {
                let d = f16_to_f32(rd16(t.d, sb * 8 + b));
                for e in 0..32 {
                    out[b * 32 + e] = (*q.add(b * 32 + e) as i8) as f32 * d;
                }
            }
        }
        NF4 => {
            for b in 0..SUPER / 64 {
                let d = f16_to_f32(rd16(t.d, sb * 4 + b));
                for e in 0..64 {
                    let i = b * 64 + e;
                    let byte = *q.add(i >> 1);
                    let idx = if i & 1 == 0 { byte & 15 } else { byte >> 4 };
                    out[i] = nf4(idx as usize) * d;
                }
            }
        }
        _ => {
            let mut dq = [0f32; 16];
            let mut mq = [0f32; 16];
            scales(t, &sp, sb, &mut dq, &mut mq);
            let qlen = 32 * sp.bits as usize;
            let off = (1u32 << (sp.bits - 1)) as f32;
            for i in 0..SUPER {
                let v = bits_at(q, qlen, i, sp.bits) as f32;
                let b = i / sp.block;
                out[i] = if sp.symmetric { (v - off) * dq[b] } else { v * dq[b] - mq[b] };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack8(p: *const u8, shuf: __m256i, shifts: __m256i, mask: __m256i) -> __m256i {
    let raw = _mm_loadl_epi64(p as *const __m128i);
    let v = _mm256_broadcastsi128_si256(raw);
    _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(v, shuf), shifts), mask)
}

/// Shuffle and shift vectors that pull 8 values of `bits` out of `bits` bytes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_consts(bits: u32) -> (__m256i, __m256i, __m256i) {
    let mut shuf = [0x80u8; 32];
    let mut sh = [0i32; 8];
    for lane in 0..8 {
        let pos = lane as u32 * bits;
        let b = (pos >> 3) as u8;
        let half = (lane / 4) * 16 + (lane % 4) * 4;
        shuf[half] = b;
        shuf[half + 1] = b + 1;
        sh[lane] = (pos & 7) as i32;
    }
    (
        _mm256_loadu_si256(shuf.as_ptr() as *const __m256i),
        _mm256_loadu_si256(sh.as_ptr() as *const __m256i),
        _mm256_set1_epi32(((1u32 << bits) - 1) as i32),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_super_avx2(t: &QTensor, sb: usize, out: &mut [f32; SUPER]) {
    let sp = spec(t.scheme).unwrap();
    let q = t.q.add(sb * 32 * sp.bits as usize);
    let o = out.as_mut_ptr();
    match t.scheme {
        Q8 => {
            for b in 0..8 {
                let d = _mm256_set1_ps(f16_to_f32(rd16(t.d, sb * 8 + b)));
                for g in 0..4 {
                    let p = q.add(b * 32 + g * 8);
                    let v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)));
                    _mm256_storeu_ps(o.add(b * 32 + g * 8), _mm256_mul_ps(v, d));
                }
            }
        }
        NF4 => {
            let lo = _mm256_loadu_ps(NF4_BITS.as_ptr() as *const f32);
            let hi = _mm256_loadu_ps(NF4_BITS.as_ptr().add(8) as *const f32);
            let (shuf, shifts, mask) = unpack_consts(4);
            for b in 0..4 {
                let d = _mm256_set1_ps(f16_to_f32(rd16(t.d, sb * 4 + b)));
                for g in 0..8 {
                    let idx = unpack8(q.add((b * 8 + g) * 4), shuf, shifts, mask);
                    let a = _mm256_permutevar8x32_ps(lo, idx);
                    let c = _mm256_permutevar8x32_ps(hi, idx);
                    // bit 3 of the index picks the upper half of the table
                    let sel = _mm256_castsi256_ps(_mm256_slli_epi32(idx, 28));
                    let lvl = _mm256_blendv_ps(a, c, sel);
                    _mm256_storeu_ps(o.add((b * 8 + g) * 8), _mm256_mul_ps(lvl, d));
                }
            }
        }
        _ => {
            let mut dq = [0f32; 16];
            let mut mq = [0f32; 16];
            scales(t, &sp, sb, &mut dq, &mut mq);
            let (shuf, shifts, mask) = unpack_consts(sp.bits);
            let off = _mm256_set1_ps((1u32 << (sp.bits - 1)) as f32);
            let groups_per_block = sp.block / 8;
            for g in 0..32 {
                let v = _mm256_cvtepi32_ps(unpack8(q.add(g * sp.bits as usize), shuf, shifts, mask));
                let b = g / groups_per_block;
                let w = if sp.symmetric {
                    _mm256_mul_ps(_mm256_sub_ps(v, off), _mm256_set1_ps(dq[b]))
                } else {
                    _mm256_sub_ps(_mm256_mul_ps(v, _mm256_set1_ps(dq[b])), _mm256_set1_ps(mq[b]))
                };
                _mm256_storeu_ps(o.add(g * 8), w);
            }
        }
    }
}

/// Dense values `[i0, i0 + len)` of an f32 / f16 / bf16 tensor as float32.
pub unsafe fn decode_dense(t: &QTensor, i0: usize, len: usize, out: *mut f32) {
    match t.scheme {
        F32 => std::ptr::copy_nonoverlapping((t.q as *const f32).add(i0), out, len),
        F16 => {
            let src = (t.q as *const u16).add(i0);
            let mut j = 0;
            #[cfg(target_arch = "x86_64")]
            if crate::simd::fast() {
                j = cvt_f16_avx(src, len, out);
            }
            for i in j..len {
                *out.add(i) = f16_to_f32(src.add(i).read_unaligned());
            }
        }
        BF16 => {
            let src = (t.q as *const u16).add(i0);
            for i in 0..len {
                *out.add(i) = f32::from_bits((src.add(i).read_unaligned() as u32) << 16);
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn cvt_f16_avx(src: *const u16, len: usize, out: *mut f32) -> usize {
    let mut i = 0;
    while i + 8 <= len {
        let h = _mm_loadu_si128(src.add(i) as *const __m128i);
        _mm256_storeu_ps(out.add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    i
}

/// Values `[k0, k0 + len)` of row `r` as float32.  For quantised schemes the
/// row width is a multiple of 256 and `k0` a multiple of 256.
pub unsafe fn decode_row(t: &QTensor, r: usize, k0: usize, len: usize, out: *mut f32) {
    let cols = t.cols as usize;
    if is_dense(t.scheme) {
        decode_dense(t, r * cols + k0, len, out);
        return;
    }
    let first = (r * cols + k0) / SUPER;
    let mut buf = [0f32; SUPER];
    let mut done = 0;
    let mut sb = first;
    while done < len {
        let take = (len - done).min(SUPER);
        if take == SUPER {
            decode_super(t, sb, &mut *(out.add(done) as *mut [f32; SUPER]));
        } else {
            decode_super(t, sb, &mut buf);
            std::ptr::copy_nonoverlapping(buf.as_ptr(), out.add(done), take);
        }
        done += take;
        sb += 1;
    }
}

/// Values `[i0, i0 + len)` of the flattened tensor as float32, in parallel – the same bits as
/// the matching slice of [`dequant`], without decoding (or holding) the rest.  Lets a caller
/// convert a large matrix by row chunks (the GPU engine uploads them one after another).
pub unsafe fn dequant_range(t: &QTensor, i0: usize, len: usize, out: *mut f32) {
    let n = t.numel();
    let len = len.min(n.saturating_sub(i0));
    if len == 0 {
        return;
    }
    let out = crate::par::Ptr(out);
    let t = *t;
    if is_dense(t.scheme) {
        crate::par::chunks(len, 1 << 16, |a, b| decode_dense(&t, i0 + a, b - a, out.get().add(a)));
        return;
    }
    let first = i0 / SUPER;
    let last = (i0 + len - 1) / SUPER;
    crate::par::chunks(last - first + 1, 256, |a, b| {
        let mut buf = [0f32; SUPER];
        for sb in first + a..first + b {
            let start = sb * SUPER;
            let lo = start.max(i0);
            let hi = (start + SUPER).min(i0 + len).min(n);
            if lo == start && hi - lo == SUPER {
                decode_super(&t, sb, &mut *(out.get().add(lo - i0) as *mut [f32; SUPER]));
            } else {
                decode_super(&t, sb, &mut buf);
                std::ptr::copy_nonoverlapping(buf.as_ptr().add(lo - start), out.get().add(lo - i0), hi - lo);
            }
        }
    });
}

/// The whole tensor as float32 (`rows · cols` values), in parallel.
pub unsafe fn dequant(t: &QTensor, out: *mut f32) {
    let n = t.numel();
    let out = crate::par::Ptr(out);
    let t = *t;
    if is_dense(t.scheme) {
        crate::par::chunks(n, 1 << 16, |a, b| decode_dense(&t, a, b - a, out.get().add(a)));
        return;
    }
    let nsup = t.supers();
    crate::par::chunks(nsup, 256, |a, b| {
        let mut buf = [0f32; SUPER];
        for sb in a..b {
            let start = sb * SUPER;
            let take = (n - start).min(SUPER);
            if take == SUPER {
                decode_super(&t, sb, &mut *(out.get().add(start) as *mut [f32; SUPER]));
            } else {
                decode_super(&t, sb, &mut buf);
                std::ptr::copy_nonoverlapping(buf.as_ptr(), out.get().add(start), take);
            }
        }
    });
}
