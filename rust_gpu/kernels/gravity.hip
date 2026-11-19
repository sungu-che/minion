// HIP kernels of the gravity-ocr GPU engine (ROCm).
//
// * gk_dequant          packed weight → float32, bit for bit what quant.dequantize computes
// * gk_gemv_<s>_<n>     y = x · Wᵀ (+ bias) for up to <n> rows of x, straight on the packed
//                       weight of scheme <s>: every lane decodes 8 values in registers and
//                       multiplies them with all rows of x; one 32-lane group per weight row
// * gk_lm_rope_cache    language: rotate adjacent pairs of q in place, rotate k, write k / v
//                       into the cache rows past..past+t
// * gk_lm_attention     language: grouped-query attention of the pass's rows over the cache
//                       (causal or block), online softmax, one block per (row, head)
// * gk_rms_norm_add     language: x += rms_norm(y) · w (the post-norm residuals)
// * gk_vis_qkv         vision: RMSNorm of the q / k heads, rotation of the two halves, the
//                       three parts split into [n, 16, 64]
// * gk_swiglu_split     u[:, I:] * silu(u[:, :I])
// * gk_silu_mul         silu(g) * u
//
// Compiled with -ffp-contract=off: the decoders and the rotations round like numpy (one
// rounding per operation); the dot products ask for fused multiply-adds explicitly (fmaf).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned char u8;

#define SUPER 256

enum Scheme { S_F32 = 0, S_F16 = 1, S_BF16 = 2, S_Q8 = 3, S_Q6 = 4, S_Q5 = 5, S_Q4 = 6, S_Q3 = 7, S_NF4 = 8 };

struct QW {
    const u8 *q;
    const u8 *d;
    const u8 *s;
    const u8 *m;
    const u8 *dm;
};

__constant__ uint32_t NF4_BITS[16] = {
    0xbf800000u, 0xbf3239b1u, 0xbf066b30u, 0xbeca32a0u, 0xbe91a24du, 0xbe3d353fu, 0xbdba7871u, 0x00000000u,
    0x3da2faffu, 0x3e24cae3u, 0x3e7c04ddu, 0x3ead033au, 0x3ee1a4b8u, 0x3f1007abu, 0x3f3913b3u, 0x3f800000u,
};

// idx / levels as numpy computes it (float32 division), as bit patterns
__constant__ uint32_t RATIO6_BITS[64] = {
    0x00000000u, 0x3c820821u, 0x3d020821u, 0x3d430c31u, 0x3d820821u, 0x3da28a29u, 0x3dc30c31u, 0x3de38e39u,
    0x3e020821u, 0x3e124925u, 0x3e228a29u, 0x3e32cb2du, 0x3e430c31u, 0x3e534d35u, 0x3e638e39u, 0x3e73cf3du,
    0x3e820821u, 0x3e8a28a3u, 0x3e924925u, 0x3e9a69a7u, 0x3ea28a29u, 0x3eaaaaabu, 0x3eb2cb2du, 0x3ebaebafu,
    0x3ec30c31u, 0x3ecb2cb3u, 0x3ed34d35u, 0x3edb6db7u, 0x3ee38e39u, 0x3eebaebbu, 0x3ef3cf3du, 0x3efbefbfu,
    0x3f020821u, 0x3f061862u, 0x3f0a28a3u, 0x3f0e38e4u, 0x3f124925u, 0x3f165966u, 0x3f1a69a7u, 0x3f1e79e8u,
    0x3f228a29u, 0x3f269a6au, 0x3f2aaaabu, 0x3f2ebaecu, 0x3f32cb2du, 0x3f36db6eu, 0x3f3aebafu, 0x3f3efbf0u,
    0x3f430c31u, 0x3f471c72u, 0x3f4b2cb3u, 0x3f4f3cf4u, 0x3f534d35u, 0x3f575d76u, 0x3f5b6db7u, 0x3f5f7df8u,
    0x3f638e39u, 0x3f679e7au, 0x3f6baebbu, 0x3f6fbefcu, 0x3f73cf3du, 0x3f77df7eu, 0x3f7befbfu, 0x3f800000u,
};
__constant__ uint32_t RATIO8_BITS[256] = {
    0x00000000u, 0x3b808081u, 0x3c008081u, 0x3c40c0c1u, 0x3c808081u, 0x3ca0a0a1u, 0x3cc0c0c1u, 0x3ce0e0e1u,
    0x3d008081u, 0x3d109091u, 0x3d20a0a1u, 0x3d30b0b1u, 0x3d40c0c1u, 0x3d50d0d1u, 0x3d60e0e1u, 0x3d70f0f1u,
    0x3d808081u, 0x3d888889u, 0x3d909091u, 0x3d989899u, 0x3da0a0a1u, 0x3da8a8a9u, 0x3db0b0b1u, 0x3db8b8b9u,
    0x3dc0c0c1u, 0x3dc8c8c9u, 0x3dd0d0d1u, 0x3dd8d8d9u, 0x3de0e0e1u, 0x3de8e8e9u, 0x3df0f0f1u, 0x3df8f8f9u,
    0x3e008081u, 0x3e048485u, 0x3e088889u, 0x3e0c8c8du, 0x3e109091u, 0x3e149495u, 0x3e189899u, 0x3e1c9c9du,
    0x3e20a0a1u, 0x3e24a4a5u, 0x3e28a8a9u, 0x3e2cacadu, 0x3e30b0b1u, 0x3e34b4b5u, 0x3e38b8b9u, 0x3e3cbcbdu,
    0x3e40c0c1u, 0x3e44c4c5u, 0x3e48c8c9u, 0x3e4ccccdu, 0x3e50d0d1u, 0x3e54d4d5u, 0x3e58d8d9u, 0x3e5cdcddu,
    0x3e60e0e1u, 0x3e64e4e5u, 0x3e68e8e9u, 0x3e6cecedu, 0x3e70f0f1u, 0x3e74f4f5u, 0x3e78f8f9u, 0x3e7cfcfdu,
    0x3e808081u, 0x3e828283u, 0x3e848485u, 0x3e868687u, 0x3e888889u, 0x3e8a8a8bu, 0x3e8c8c8du, 0x3e8e8e8fu,
    0x3e909091u, 0x3e929293u, 0x3e949495u, 0x3e969697u, 0x3e989899u, 0x3e9a9a9bu, 0x3e9c9c9du, 0x3e9e9e9fu,
    0x3ea0a0a1u, 0x3ea2a2a3u, 0x3ea4a4a5u, 0x3ea6a6a7u, 0x3ea8a8a9u, 0x3eaaaaabu, 0x3eacacadu, 0x3eaeaeafu,
    0x3eb0b0b1u, 0x3eb2b2b3u, 0x3eb4b4b5u, 0x3eb6b6b7u, 0x3eb8b8b9u, 0x3ebababbu, 0x3ebcbcbdu, 0x3ebebebfu,
    0x3ec0c0c1u, 0x3ec2c2c3u, 0x3ec4c4c5u, 0x3ec6c6c7u, 0x3ec8c8c9u, 0x3ecacacbu, 0x3ecccccdu, 0x3ecececfu,
    0x3ed0d0d1u, 0x3ed2d2d3u, 0x3ed4d4d5u, 0x3ed6d6d7u, 0x3ed8d8d9u, 0x3edadadbu, 0x3edcdcddu, 0x3edededfu,
    0x3ee0e0e1u, 0x3ee2e2e3u, 0x3ee4e4e5u, 0x3ee6e6e7u, 0x3ee8e8e9u, 0x3eeaeaebu, 0x3eececedu, 0x3eeeeeefu,
    0x3ef0f0f1u, 0x3ef2f2f3u, 0x3ef4f4f5u, 0x3ef6f6f7u, 0x3ef8f8f9u, 0x3efafafbu, 0x3efcfcfdu, 0x3efefeffu,
    0x3f008081u, 0x3f018182u, 0x3f028283u, 0x3f038384u, 0x3f048485u, 0x3f058586u, 0x3f068687u, 0x3f078788u,
    0x3f088889u, 0x3f09898au, 0x3f0a8a8bu, 0x3f0b8b8cu, 0x3f0c8c8du, 0x3f0d8d8eu, 0x3f0e8e8fu, 0x3f0f8f90u,
    0x3f109091u, 0x3f119192u, 0x3f129293u, 0x3f139394u, 0x3f149495u, 0x3f159596u, 0x3f169697u, 0x3f179798u,
    0x3f189899u, 0x3f19999au, 0x3f1a9a9bu, 0x3f1b9b9cu, 0x3f1c9c9du, 0x3f1d9d9eu, 0x3f1e9e9fu, 0x3f1f9fa0u,
    0x3f20a0a1u, 0x3f21a1a2u, 0x3f22a2a3u, 0x3f23a3a4u, 0x3f24a4a5u, 0x3f25a5a6u, 0x3f26a6a7u, 0x3f27a7a8u,
    0x3f28a8a9u, 0x3f29a9aau, 0x3f2aaaabu, 0x3f2babacu, 0x3f2cacadu, 0x3f2dadaeu, 0x3f2eaeafu, 0x3f2fafb0u,
    0x3f30b0b1u, 0x3f31b1b2u, 0x3f32b2b3u, 0x3f33b3b4u, 0x3f34b4b5u, 0x3f35b5b6u, 0x3f36b6b7u, 0x3f37b7b8u,
    0x3f38b8b9u, 0x3f39b9bau, 0x3f3ababbu, 0x3f3bbbbcu, 0x3f3cbcbdu, 0x3f3dbdbeu, 0x3f3ebebfu, 0x3f3fbfc0u,
    0x3f40c0c1u, 0x3f41c1c2u, 0x3f42c2c3u, 0x3f43c3c4u, 0x3f44c4c5u, 0x3f45c5c6u, 0x3f46c6c7u, 0x3f47c7c8u,
    0x3f48c8c9u, 0x3f49c9cau, 0x3f4acacbu, 0x3f4bcbccu, 0x3f4ccccdu, 0x3f4dcdceu, 0x3f4ececfu, 0x3f4fcfd0u,
    0x3f50d0d1u, 0x3f51d1d2u, 0x3f52d2d3u, 0x3f53d3d4u, 0x3f54d4d5u, 0x3f55d5d6u, 0x3f56d6d7u, 0x3f57d7d8u,
    0x3f58d8d9u, 0x3f59d9dau, 0x3f5adadbu, 0x3f5bdbdcu, 0x3f5cdcddu, 0x3f5ddddeu, 0x3f5ededfu, 0x3f5fdfe0u,
    0x3f60e0e1u, 0x3f61e1e2u, 0x3f62e2e3u, 0x3f63e3e4u, 0x3f64e4e5u, 0x3f65e5e6u, 0x3f66e6e7u, 0x3f67e7e8u,
    0x3f68e8e9u, 0x3f69e9eau, 0x3f6aeaebu, 0x3f6bebecu, 0x3f6cecedu, 0x3f6dedeeu, 0x3f6eeeefu, 0x3f6feff0u,
    0x3f70f0f1u, 0x3f71f1f2u, 0x3f72f2f3u, 0x3f73f3f4u, 0x3f74f4f5u, 0x3f75f5f6u, 0x3f76f6f7u, 0x3f77f7f8u,
    0x3f78f8f9u, 0x3f79f9fau, 0x3f7afafbu, 0x3f7bfbfcu, 0x3f7cfcfdu, 0x3f7dfdfeu, 0x3f7efeffu, 0x3f800000u,
};

__device__ __forceinline__ uint32_t rd16(const u8 *p, size_t i) {
    return (uint32_t)p[2 * i] | ((uint32_t)p[2 * i + 1] << 8);
}

// exact IEEE half → float (the hardware conversion keeps subnormals)
__device__ __forceinline__ float h2f(uint32_t h) { return __half2float(__ushort_as_half((unsigned short)h)); }

// the same in integer arithmetic (reference for the self-check of h2f)
__device__ __forceinline__ float h2f_bits(uint32_t h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int ex = 113;
            while (!(m & 0x400u)) {
                m <<= 1;
                ex--;
            }
            bits = s | ((uint32_t)ex << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    return __uint_as_float(bits);
}

// value i of a little-endian stream of `bits`-wide values
__device__ __forceinline__ uint32_t bits_at(const u8 *p, size_t i, int bits) {
    const size_t pos = i * (size_t)bits;
    const size_t byte = pos >> 3;
    const int shift = (int)(pos & 7);
    uint32_t v = p[byte];
    if (shift + bits > 8) v |= (uint32_t)p[byte + 1] << 8;
    return (v >> shift) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float ratio(uint32_t idx, uint32_t levels) {
    return __uint_as_float(levels == 63u ? RATIO6_BITS[idx] : RATIO8_BITS[idx]);
}

template <int BITS, int BLOCK, bool SYM, int SBITS>
__device__ __forceinline__ float deq_k(const QW &w, size_t i) {
    const size_t blk = i / BLOCK;
    const size_t sup = i / SUPER;
    const uint32_t levels = (1u << SBITS) - 1u;
    const float dq = __fmul_rn(h2f(rd16(w.d, sup)), ratio(bits_at(w.s, blk, SBITS), levels));
    const float q = (float)bits_at(w.q, i, BITS);
    if (SYM) return __fmul_rn(__fsub_rn(q, (float)(1 << (BITS - 1))), dq);
    const float mq = __fmul_rn(h2f(rd16(w.dm, sup)), ratio(bits_at(w.m, blk, SBITS), levels));
    return __fsub_rn(__fmul_rn(q, dq), mq);
}

template <int S>
__device__ __forceinline__ float deq(const QW &w, size_t i) {
    if (S == S_F32) return ((const float *)w.q)[i];
    if (S == S_F16) return h2f(rd16(w.q, i));
    if (S == S_BF16) return __uint_as_float(rd16(w.q, i) << 16);
    if (S == S_Q8) return __fmul_rn((float)(int8_t)w.q[i], h2f(rd16(w.d, i / 32)));
    if (S == S_NF4) {
        const uint32_t b = w.q[i >> 1];
        const uint32_t idx = (i & 1) ? (b >> 4) : (b & 15u);
        return __fmul_rn(__uint_as_float(NF4_BITS[idx]), h2f(rd16(w.d, i / 64)));
    }
    if (S == S_Q6) return deq_k<6, 16, true, 8>(w, i);
    if (S == S_Q5) return deq_k<5, 32, false, 6>(w, i);
    if (S == S_Q4) return deq_k<4, 32, false, 6>(w, i);
    return deq_k<3, 16, true, 6>(w, i);  // S_Q3
}

// ---------------------------------------------------------------------------------------------
// dequantisation
// ---------------------------------------------------------------------------------------------
template <int S>
__device__ void dequant_impl(const QW w, float *out, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = deq<S>(w, i);
}

#define DEQ_FN(S, NAME)                                                                               \
    extern "C" __global__ void __launch_bounds__(256) gk_dequant_##NAME(QW w, float *out, size_t n) { \
        dequant_impl<S>(w, out, n);                                                                   \
    }
DEQ_FN(S_F32, f32)
DEQ_FN(S_F16, f16)
DEQ_FN(S_BF16, bf16)
DEQ_FN(S_Q8, q8)
DEQ_FN(S_Q6, q6)
DEQ_FN(S_Q5, q5)
DEQ_FN(S_Q4, q4)
DEQ_FN(S_Q3, q3)
DEQ_FN(S_NF4, nf4)

// ---------------------------------------------------------------------------------------------
// fused decode + multiply for a few rows of x
// ---------------------------------------------------------------------------------------------
#define LANES 32
#define ROWS_PER_BLOCK 8

// The 8 values of group g (values 8g .. 8g+7) of super-block sb.  A group never straddles a
// block, so its scale is decoded once; the values come from `bits` consecutive bytes.
template <int BITS, int BLOCK, bool SYM, int SBITS>
__device__ __forceinline__ void group8_k(const QW &w, size_t i0, float v[8]) {
    const size_t blk = i0 / BLOCK;
    const size_t sup = i0 / SUPER;
    const uint32_t levels = (1u << SBITS) - 1u;
    const float dq = __fmul_rn(h2f(rd16(w.d, sup)), ratio(bits_at(w.s, blk, SBITS), levels));
    float mq = 0.0f;
    if (!SYM) mq = __fmul_rn(h2f(rd16(w.dm, sup)), ratio(bits_at(w.m, blk, SBITS), levels));
    const u8 *p = w.q + i0 / 8 * BITS;
    uint64_t bits = 0;
#pragma unroll
    for (int b = 0; b < BITS; ++b) bits |= (uint64_t)p[b] << (8 * b);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float q = (float)((bits >> (e * BITS)) & ((1u << BITS) - 1u));
        v[e] = SYM ? __fmul_rn(__fsub_rn(q, (float)(1 << (BITS - 1))), dq) : __fsub_rn(__fmul_rn(q, dq), mq);
    }
}

template <int S>
__device__ __forceinline__ void group8(const QW &w, size_t sb, int g, float v[8]) {
    const size_t i0 = sb * SUPER + (size_t)g * 8;
    if (S == S_F32) {
        const float4 a = *(const float4 *)(w.q + 4 * i0);
        const float4 b = *(const float4 *)(w.q + 4 * i0 + 16);
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    } else if (S == S_F16 || S == S_BF16) {
        const uint4 h = *(const uint4 *)(w.q + 2 * i0);
        const uint32_t hw[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const uint32_t lo = hw[e] & 0xffffu, hi = hw[e] >> 16;
            v[2 * e] = S == S_F16 ? h2f(lo) : __uint_as_float(lo << 16);
            v[2 * e + 1] = S == S_F16 ? h2f(hi) : __uint_as_float(hi << 16);
        }
    } else if (S == S_Q8) {
        const float d = h2f(rd16(w.d, i0 / 32));
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = __fmul_rn((float)(int8_t)w.q[i0 + e], d);
    } else if (S == S_NF4) {
        const float d = h2f(rd16(w.d, i0 / 64));
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const uint32_t b = w.q[i0 / 2 + e];
            v[2 * e] = __fmul_rn(__uint_as_float(NF4_BITS[b & 15u]), d);
            v[2 * e + 1] = __fmul_rn(__uint_as_float(NF4_BITS[b >> 4]), d);
        }
    } else if (S == S_Q6) {
        group8_k<6, 16, true, 8>(w, i0, v);
    } else if (S == S_Q5) {
        group8_k<5, 32, false, 6>(w, i0, v);
    } else if (S == S_Q4) {
        group8_k<4, 32, false, 6>(w, i0, v);
    } else {
        group8_k<3, 16, true, 6>(w, i0, v);
    }
}

template <int S, int NX>
__device__ void gemv_impl(const QW w, const float *__restrict__ x, const float *__restrict__ bias,
                          float *__restrict__ y, int rows, int cols, int n, int accumulate) {
    const int lane = threadIdx.x % LANES;
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / LANES;
    float acc[NX];
#pragma unroll
    for (int j = 0; j < NX; ++j) acc[j] = 0.0f;
    if (r < rows) {
        const int supers = cols / SUPER;
        const size_t row_sb = (size_t)r * supers;
        for (int k = 0; k < supers; ++k) {
            float v[8];
            group8<S>(w, row_sb + k, lane, v);
            const int c0 = k * SUPER + lane * 8;
#pragma unroll
            for (int j = 0; j < NX; ++j) {
                if (j < n) {
                    const float4 a = *(const float4 *)(x + (size_t)j * cols + c0);
                    const float4 b = *(const float4 *)(x + (size_t)j * cols + c0 + 4);
                    float s = acc[j];
                    s = fmaf(v[0], a.x, s);
                    s = fmaf(v[1], a.y, s);
                    s = fmaf(v[2], a.z, s);
                    s = fmaf(v[3], a.w, s);
                    s = fmaf(v[4], b.x, s);
                    s = fmaf(v[5], b.y, s);
                    s = fmaf(v[6], b.z, s);
                    s = fmaf(v[7], b.w, s);
                    acc[j] = s;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NX; ++j) {
        float s = acc[j];
#pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1) s += __shfl_xor(s, off, LANES);
        acc[j] = s;
    }
    if (r < rows && lane == 0) {
        const float b = bias ? bias[r] : 0.0f;
        for (int j = 0; j < n && j < NX; ++j) {
            float *dst = y + (size_t)j * rows + r;
            const float val = acc[j] + b;
            *dst = accumulate ? *dst + val : val;
        }
    }
}

#define GEMV_FN(S, NAME, NX)                                                                                  \
    extern "C" __global__ void __launch_bounds__(LANES * ROWS_PER_BLOCK)                                      \
        gk_gemv_##NAME##_##NX(QW w, const float *x, const float *bias, float *y, int rows, int cols, int n, \
                              int accumulate) {                                                           \
        gemv_impl<S, NX>(w, x, bias, y, rows, cols, n, accumulate);                                         \
    }
#define GEMV_ALL(S, NAME) \
    GEMV_FN(S, NAME, 1)   \
    GEMV_FN(S, NAME, 2)   \
    GEMV_FN(S, NAME, 4)   \
    GEMV_FN(S, NAME, 8)   \
    GEMV_FN(S, NAME, 16)  \
    GEMV_FN(S, NAME, 32)
GEMV_ALL(S_F32, f32)
GEMV_ALL(S_F16, f16)
GEMV_ALL(S_BF16, bf16)
GEMV_ALL(S_Q8, q8)
GEMV_ALL(S_Q6, q6)
GEMV_ALL(S_Q5, q5)
GEMV_ALL(S_Q4, q4)
GEMV_ALL(S_Q3, q3)
GEMV_ALL(S_NF4, nf4)

// ---------------------------------------------------------------------------------------------
// fused decode + GEMM, tiled in shared memory
// Y[n, rows] (+)= X[n, cols] · Wᵀ (+ bias).  A block of 256 threads owns a BM × 64 tile of Y; for
// every chunk of 32 columns it loads the X tile and decodes the W tile (one 8-value group per
// thread, scale decoded once per group) into shared memory, then every thread accumulates
// (BM/16) × 4 outputs with fused multiply-adds.  cols % 256 == 0.
// ---------------------------------------------------------------------------------------------
#define QG_BN 64
#define QG_BK 32
#define QG_THREADS 256

template <int S, int BM>
__device__ void qgemm_impl(const QW w, const float *__restrict__ x, const float *__restrict__ bias,
                           float *__restrict__ y, int rows, int cols, int n, int accumulate) {
    constexpr int TM = BM / 16;
    __shared__ float As[QG_BK][BM + 1];
    __shared__ float Bs[QG_BK][QG_BN + 1];
    const int tid = threadIdx.x;
    const int tx = tid % 16;  // x rows tx + 16 i
    const int ty = tid / 16;  // w rows ty + 16 j
    const int m0 = blockIdx.y * BM;
    const int r0 = blockIdx.x * QG_BN;
    float acc[TM][4];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    const int supers = cols / SUPER;
    for (int k0 = 0; k0 < cols; k0 += QG_BK) {
        // X tile: BM rows × 32 columns, 8 consecutive values per thread
        for (int e = tid; e < BM * (QG_BK / 8); e += QG_THREADS) {
            const int mm = e / (QG_BK / 8);
            const int kk = (e % (QG_BK / 8)) * 8;
            const int m = m0 + mm;
            float v[8];
            if (m < n) {
                const float4 a = *(const float4 *)(x + (size_t)m * cols + k0 + kk);
                const float4 b = *(const float4 *)(x + (size_t)m * cols + k0 + kk + 4);
                v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) v[q] = 0.0f;
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) As[kk + q][mm] = v[q];
        }
        // W tile: 64 rows × 32 columns = 256 groups of 8, one per thread
        {
            const int rr = tid / (QG_BK / 8);
            const int gg = tid % (QG_BK / 8);
            const int r = r0 + rr;
            float v[8];
            if (r < rows) {
                const size_t sb = (size_t)r * supers + k0 / SUPER;
                group8<S>(w, sb, (k0 % SUPER) / 8 + gg, v);
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) v[q] = 0.0f;
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) Bs[gg * 8 + q][rr] = v[q];
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < QG_BK; ++kk) {
            float a[TM], b[4];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][tx + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][ty + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = r0 + ty + 16 * j;
        if (r >= rows) continue;
        const float bb = bias ? bias[r] : 0.0f;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = m0 + tx + 16 * i;
            if (m < n) {
                float *dst = y + (size_t)m * rows + r;
                const float val = acc[i][j] + bb;
                *dst = accumulate ? *dst + val : val;
            }
        }
    }
}

#define QGEMM_FN(S, NAME, BM)                                                                                 \
    extern "C" __global__ void __launch_bounds__(QG_THREADS)                                                  \
        gk_qgemm_##NAME##_##BM(QW w, const float *x, const float *bias, float *y, int rows, int cols, int n, \
                               int accumulate) {                                                            \
        qgemm_impl<S, BM>(w, x, bias, y, rows, cols, n, accumulate);                                          \
    }
#define QGEMM_ALL(S, NAME) \
    QGEMM_FN(S, NAME, 32)  \
    QGEMM_FN(S, NAME, 64)
QGEMM_ALL(S_F32, f32)
QGEMM_ALL(S_F16, f16)
QGEMM_ALL(S_BF16, bf16)
QGEMM_ALL(S_Q8, q8)
QGEMM_ALL(S_Q6, q6)
QGEMM_ALL(S_Q5, q5)
QGEMM_ALL(S_Q4, q4)
QGEMM_ALL(S_Q3, q3)
QGEMM_ALL(S_NF4, nf4)

// ---------------------------------------------------------------------------------------------
// language: rotary embedding of adjacent pairs + cache write
// q [t, 16, 128] (rotated in place), k / v [t, 8, 128], cos / sin [t, 64], kc / vc [cap, 8, 128]
// one block of 64 threads per (token, head row); rows 0..15 q, 16..23 k, 24..31 v
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64)
    gk_lm_rope_cache(float *q, const float *k, const float *v, const float *cos, const float *sin, float *kc,
                     float *vc, int t, int past) {
    const int row = blockIdx.x;
    const int i = row / 32;
    const int h = row % 32;
    const int p = threadIdx.x;  // pair index 0..63
    if (i >= t) return;
    const float c = cos[i * 64 + p];
    const float s = sin[i * 64 + p];
    if (h < 16) {
        float *x = q + ((size_t)i * 16 + h) * 128;
        const float a = x[2 * p], b = x[2 * p + 1];
        x[2 * p] = a * c - b * s;
        x[2 * p + 1] = b * c + a * s;
    } else if (h < 24) {
        const int g = h - 16;
        const float *x = k + ((size_t)i * 8 + g) * 128;
        float *dst = kc + ((size_t)(past + i) * 8 + g) * 128;
        const float a = x[2 * p], b = x[2 * p + 1];
        dst[2 * p] = a * c - b * s;
        dst[2 * p + 1] = b * c + a * s;
    } else {
        const int g = h - 24;
        const float *x = v + ((size_t)i * 8 + g) * 128;
        float *dst = vc + ((size_t)(past + i) * 8 + g) * 128;
        dst[2 * p] = x[2 * p];
        dst[2 * p + 1] = x[2 * p + 1];
    }
}

// ---------------------------------------------------------------------------------------------
// language attention over the cache (the few query rows of a decoding pass)
// q [t, 16, 128], kc / vc [cap, 8, 128] (rows 0..past+t written), out [t, 16, 128]
// one block of 128 threads (one per head dimension) per (query row, head); heads 2g and 2g+1 read
// KV head g; causal = bottom-right aligned (row i sees the keys 0..past+i); the keys are taken in
// chunks of LM_ATT_CHUNK with an online softmax, f32 throughout
// ---------------------------------------------------------------------------------------------
#define LM_ATT_CHUNK 1024
extern "C" __global__ void __launch_bounds__(128)
    gk_lm_attention(const float *q, const float *kc, const float *vc, float *out, int t, int past, int causal,
                    float scale) {
    __shared__ float qs[128];
    __shared__ float sc[LM_ATT_CHUNK];
    __shared__ float red[128];
    const int row = blockIdx.x / 16, h = blockIdx.x % 16, g = h / 2, tid = threadIdx.x;
    if (row >= t) return;
    const int n = causal ? past + row + 1 : past + t;
    qs[tid] = q[((size_t)row * 16 + h) * 128 + tid] * scale;
    __syncthreads();
    float m = -INFINITY, l = 0.0f, acc = 0.0f;  // running max, running sum, output of dimension tid
    for (int c0 = 0; c0 < n; c0 += LM_ATT_CHUNK) {
        const int len = min(LM_ATT_CHUNK, n - c0);
        float cmax = -INFINITY;
        for (int j = tid; j < len; j += 128) {
            const float4 *k = (const float4 *)(kc + ((size_t)(c0 + j) * 8 + g) * 128);
            float s = 0.0f;
            for (int d = 0; d < 32; ++d) {
                const float4 kv = k[d];
                s = fmaf(qs[4 * d], kv.x, s);
                s = fmaf(qs[4 * d + 1], kv.y, s);
                s = fmaf(qs[4 * d + 2], kv.z, s);
                s = fmaf(qs[4 * d + 3], kv.w, s);
            }
            sc[j] = s;
            cmax = fmaxf(cmax, s);
        }
        red[tid] = cmax;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        const float mnew = fmaxf(m, red[0]);
        __syncthreads();
        float csum = 0.0f;
        for (int j = tid; j < len; j += 128) {
            const float p = expf(sc[j] - mnew);
            sc[j] = p;
            csum += p;
        }
        red[tid] = csum;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float alpha = expf(m - mnew);  // 0 on the first chunk (m = -inf)
        l = l * alpha + red[0];
        acc *= alpha;
        const float *v = vc + ((size_t)c0 * 8 + g) * 128 + tid;
        for (int j = 0; j < len; ++j) acc = fmaf(sc[j], v[(size_t)j * 8 * 128], acc);
        m = mnew;
        __syncthreads();
    }
    out[((size_t)row * 16 + h) * 128 + tid] = acc / l;
}

// ---------------------------------------------------------------------------------------------
// the same attention, tiled: one block per (KV head g, tile of LMT_ROWS query rows) holds the
// 2 · LMT_ROWS query vectors that read KV head g (heads 2g, 2g+1 of every row of the tile), so a
// chunk of LMT_KEYS keys is loaded into LDS once for all of them; 128 threads, thread d owns output
// dimension d of every query; online softmax per query over the chunks
// ---------------------------------------------------------------------------------------------
#define LMT_ROWS 8
#define LMT_Q (2 * LMT_ROWS)
#define LMT_KEYS 32
extern "C" __global__ void __launch_bounds__(128)
    gk_lm_attention_tiled(const float *q, const float *kc, const float *vc, float *out, int t, int past, int causal,
                          float scale) {
    __shared__ float qs[LMT_Q][128];
    __shared__ float ks[LMT_KEYS][128 + 1];  // +1: no bank conflicts when a thread walks a key row
    __shared__ float ps[LMT_Q][LMT_KEYS];
    __shared__ float mrun[LMT_Q], lrun[LMT_Q], alpha[LMT_Q];
    const int g = blockIdx.x % 8, tile = blockIdx.x / 8, tid = threadIdx.x;
    const int row0 = tile * LMT_ROWS;
    const int rows = min(LMT_ROWS, t - row0);
    if (rows <= 0) return;
    for (int i = 0; i < LMT_Q; ++i) {  // query i = (row row0 + i / 2, head 2g + i % 2)
        const int r = row0 + i / 2;
        qs[i][tid] = r < t ? q[((size_t)r * 16 + 2 * g + (i & 1)) * 128 + tid] * scale : 0.0f;
    }
    if (tid < LMT_Q) {
        mrun[tid] = -INFINITY;
        lrun[tid] = 0.0f;
    }
    // keys any row of the tile may see
    const int n = causal ? past + row0 + rows : past + t;
    float acc[LMT_Q];
    for (int i = 0; i < LMT_Q; ++i) acc[i] = 0.0f;
    __syncthreads();
    for (int c0 = 0; c0 < n; c0 += LMT_KEYS) {
        const int len = min(LMT_KEYS, n - c0);
        for (int e = tid; e < len * 128; e += 128) {
            const int j = e / 128, d = e % 128;
            ks[j][d] = kc[((size_t)(c0 + j) * 8 + g) * 128 + d];
        }
        __syncthreads();
        // scores: LMT_Q x len dot products, 4 per thread
        for (int s = tid; s < LMT_Q * LMT_KEYS; s += 128) {
            const int i = s / LMT_KEYS, j = s % LMT_KEYS;
            const int r = row0 + i / 2;
            const int limit = causal ? past + r + 1 : past + t;  // keys 0..limit-1 visible to row r
            float v = -INFINITY;
            if (j < len && r < t && c0 + j < limit) {
                v = 0.0f;
                for (int d = 0; d < 128; ++d) v = fmaf(qs[i][d], ks[j][d], v);
            }
            ps[i][j] = v;
        }
        __syncthreads();
        // per query: new running max, the chunk's exponentials, the rescale of what came before
        if (tid < LMT_Q) {
            float cm = -INFINITY;
            for (int j = 0; j < len; ++j) cm = fmaxf(cm, ps[tid][j]);
            const float mnew = fmaxf(mrun[tid], cm);
            float sum = 0.0f;
            for (int j = 0; j < len; ++j) {
                const float p = mnew == -INFINITY ? 0.0f : expf(ps[tid][j] - mnew);
                ps[tid][j] = p;
                sum += p;
            }
            const float a = mrun[tid] == -INFINITY ? 0.0f : expf(mrun[tid] - mnew);
            alpha[tid] = a;
            lrun[tid] = lrun[tid] * a + sum;
            mrun[tid] = mnew;
        }
        __syncthreads();
        for (int i = 0; i < LMT_Q; ++i) acc[i] *= alpha[i];
        const float *v = vc + ((size_t)c0 * 8 + g) * 128 + tid;
        for (int j = 0; j < len; ++j) {
            const float vj = v[(size_t)j * 8 * 128];
            for (int i = 0; i < LMT_Q; ++i) acc[i] = fmaf(ps[i][j], vj, acc[i]);
        }
        __syncthreads();
    }
    for (int i = 0; i < LMT_Q; ++i) {
        const int r = row0 + i / 2;
        if (r < t) out[((size_t)r * 16 + 2 * g + (i & 1)) * 128 + tid] = acc[i] / lrun[i];
    }
}

// ---------------------------------------------------------------------------------------------
// vision: qkv [n, 3, 16, 64] → q, k (RMSNorm per head + rotation of the halves), v; each [n, 16, 64]
// one block of 64 threads per (token, head, part)
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64)
    gk_vis_qkv(const float *qkv, const float *qn, const float *kn, const float *cos, const float *sin, float eps,
               float *q, float *k, float *v, int n) {
    __shared__ float red[64];
    __shared__ float hrow[64];
    const int b = blockIdx.x;
    const int part = b % 3;
    const int head = (b / 3) % 16;
    const int i = b / 48;
    const int d = threadIdx.x;
    if (i >= n) return;
    const float x = qkv[((size_t)i * 3 + part) * 1024 + head * 64 + d];
    const size_t o = ((size_t)i * 16 + head) * 64 + d;
    if (part == 2) {
        v[o] = x;
        return;
    }
    red[d] = x * x;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if (d < off) red[d] += red[d + off];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(red[0] / 64.0f + eps);
    const float hv = x * scale * (part == 0 ? qn[d] : kn[d]);
    hrow[d] = hv;
    __syncthreads();
    const float rot = d < 32 ? -hrow[d + 32] : hrow[d - 32];
    const float out = hv * cos[(size_t)i * 64 + d] + rot * sin[(size_t)i * 64 + d];
    (part == 0 ? q : k)[o] = out;
}

// ---------------------------------------------------------------------------------------------
// post-norm residual of the language layers: x[i] += rms_norm(y[i]) * w, one block per row
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
    gk_rms_norm_add(float *x, const float *y, const float *w, float eps, int d) {
    __shared__ float red[256];
    const size_t row = blockIdx.x;
    const float *yr = y + row * (size_t)d;
    float *xr = x + row * (size_t)d;
    float s = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s = fmaf(yr[j], yr[j], s);
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(red[0] / (float)d + eps);
    for (int j = threadIdx.x; j < d; j += blockDim.x) xr[j] += yr[j] * scale * w[j];
}

// ---------------------------------------------------------------------------------------------
// gated activations
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

extern "C" __global__ void __launch_bounds__(256)
    gk_swiglu_split(const float *u, float *out, size_t n, size_t inter) {
    const size_t total = n * inter;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride) {
        const size_t i = e / inter, j = e % inter;
        const float g = u[i * 2 * inter + j];
        out[e] = u[i * 2 * inter + inter + j] * silu(g);
    }
}

extern "C" __global__ void __launch_bounds__(256) gk_silu_mul(const float *g, const float *u, float *out, size_t len) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < len; e += stride) out[e] = silu(g[e]) * u[e];
}

extern "C" __global__ void gk_probe(float *x) { x[threadIdx.x] += 1.0f; }
