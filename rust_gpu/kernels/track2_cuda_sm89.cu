// Track 2 (GLM-OCR) Native BF16 CUDA (sm_89 / Ada Lovelace L40S) Fused Kernels.
// Upgraded from rust_gpu/kernels/gravity.hip to native __nv_bfloat16 + warp-shuffle intrinsics
// for zero-copy PyTorch data_ptr() integration on torch.cuda.current_stream() (100% CUDA Graph safe).
//
// Build inside track2-base:v1 (no --use_fast_math to preserve exact IEEE sqrtf/RoPE precision):
//   nvcc -O3 -std=c++17 -arch=sm_89 --shared -Xcompiler -fPIC \
//        rust_gpu/kernels/track2_cuda_sm89.cu -o gravity_ocr/_native/libtrack2_cuda_sm89.so

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------------------------
// 1. gk_u8_norm_bf16: [N, 588] uint8 (3 x 14 x 14 per patch) -> [N, 588] __nv_bfloat16
//    Single-frame normalized patch tensor for folded GlmOcrVisionPatchEmbed (4x smaller IPC/H2D).
//    Channel layout inside each 588-element row:
//      [0..195]   = R (c=0)
//      [196..391] = G (c=1)
//      [392..587] = B (c=2)
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
gk_u8_norm_bf16_kernel(
    const uint8_t* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n_patches,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2
) {
    const int64_t total = n_patches * 588;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        const int col = (int)(i % 588);
        const float v = (float)src[i];
        float out;
        if (col < 196) {
            out = fmaf(v, s0, b0);
        } else if (col < 392) {
            out = fmaf(v, s1, b1);
        } else {
            out = fmaf(v, s2, b2);
        }
        dst[i] = __float2bfloat16_rn(out);
    }
}

// ---------------------------------------------------------------------------------------------
// 1B. gk_raw_rgb_patchify_norm_bf16: Raw HWC uint8 RGB [gh*14, gw*14, 3] (passed as [gh*gw, 588] uint8)
//     -> 2x2 merge-group ordered + normalized [gh*gw, 588] __nv_bfloat16 in ONE GPU pass!
//     Eliminates CPU patchify, eliminates CPU float->bf16, cuts ZMQ IPC & PCIe H2D by 4x (216MB vs 865MB),
//     and feeds 2x smaller folded GlmOcrVisionPatchEmbed F.linear(x, W0 + W1, bias).
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
gk_raw_rgb_patchify_norm_bf16_kernel(
    const uint8_t* __restrict__ rgb_hwc,
    __nv_bfloat16* __restrict__ dst_patches,
    int gh,
    int gw,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2
) {
    const int n_patches = gh * gw;
    const int64_t total = (int64_t)n_patches * 588;
    const int gw2 = gw >> 1;
    const int w_px = gw * 14;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t t = idx; t < total; t += stride) {
        const int p = (int)(t / 588);
        const int col = (int)(t % 588);

        const int group_idx = p >> 2;
        const int sub_idx = p & 3;
        const int gr = group_idx / gw2;
        const int gc = group_idx % gw2;
        const int pr = (gr << 1) + (sub_idx >> 1);
        const int pc = (gc << 1) + (sub_idx & 1);

        int c, rem;
        float sc, bi;
        if (col < 196) {
            c = 0; rem = col; sc = s0; bi = b0;
        } else if (col < 392) {
            c = 1; rem = col - 196; sc = s1; bi = b1;
        } else {
            c = 2; rem = col - 392; sc = s2; bi = b2;
        }
        const int py = rem / 14;
        const int px = rem % 14;

        const int y = pr * 14 + py;
        const int x = pc * 14 + px;
        const uint8_t v = rgb_hwc[((int64_t)y * w_px + x) * 3 + c];
        dst_patches[t] = __float2bfloat16_rn(fmaf((float)v, sc, bi));
    }
}

// ---------------------------------------------------------------------------------------------
// 1C. gk_chw_u8_patchify_norm_bf16: Resized CHW uint8 RGB [3, gh*14, gw*14] on GPU
//     (direct output of torchvision GPU resize!) -> 2x2 merge-group ordered + normalized
//     [gh*gw, 588] __nv_bfloat16 in ONE 8us GPU pass!
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
gk_chw_u8_patchify_norm_bf16_kernel(
    const uint8_t* __restrict__ rgb_chw,
    __nv_bfloat16* __restrict__ dst_patches,
    int gh,
    int gw,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2
) {
    const int n_patches = gh * gw;
    const int64_t total = (int64_t)n_patches * 588;
    const int gw2 = gw >> 1;
    const int h_px = gh * 14;
    const int w_px = gw * 14;
    const int64_t plane = (int64_t)h_px * w_px;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t t = idx; t < total; t += stride) {
        const int p = (int)(t / 588);
        const int col = (int)(t % 588);

        const int group_idx = p >> 2;
        const int sub_idx = p & 3;
        const int gr = group_idx / gw2;
        const int gc = group_idx % gw2;
        const int pr = (gr << 1) + (sub_idx >> 1);
        const int pc = (gc << 1) + (sub_idx & 1);

        int c, rem;
        float sc, bi;
        if (col < 196) {
            c = 0; rem = col; sc = s0; bi = b0;
        } else if (col < 392) {
            c = 1; rem = col - 196; sc = s1; bi = b1;
        } else {
            c = 2; rem = col - 392; sc = s2; bi = b2;
        }
        const int py = rem / 14;
        const int px = rem % 14;

        const int y = pr * 14 + py;
        const int x = pc * 14 + px;
        const uint8_t v = rgb_chw[(int64_t)c * plane + (int64_t)y * w_px + x];
        dst_patches[t] = __float2bfloat16_rn(fmaf((float)v, sc, bi));
    }
}

// ---------------------------------------------------------------------------------------------
// 2. gk_vis_qkv_sdpa_bf16: Fused 9-in-1 CogViT Attention Prep (Upgraded from gravity.hip gk_vis_qkv)
//    Input:
//      qkv [S, 3, H, 64] (__nv_bfloat16)  (H=16, D=64, total last dim=3072)
//      qn  [64] (__nv_bfloat16)           (q_norm RMSNorm weight)
//      kn  [64] (__nv_bfloat16)           (k_norm RMSNorm weight)
//      cos [S, 64] (float32 or bf16 -> fp32 passed via flag)
//      sin [S, 64]
//    Output (directly in [H, S, 64] contiguous SDPA layout!):
//      q_out [H, S, 64] (__nv_bfloat16)
//      k_out [H, S, 64] (__nv_bfloat16)
//      v_out [H, S, 64] (__nv_bfloat16)
//
//    Zero __shared__ memory, zero __syncthreads():
//      1 warp (32 threads) per (token s, head h, part p in {0=Q, 1=K, 2=V}).
//      Lane d (0..31) owns BOTH (d, d+32) in registers: exact NeoX rotation pair!
//      RMSNorm sum-of-squares uses 5-step warp __shfl_xor_sync reduction.
// ---------------------------------------------------------------------------------------------
template <bool COS_BF16, bool OUT_SHD>
__global__ void __launch_bounds__(128)
gk_vis_qkv_sdpa_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qn,
    const __nv_bfloat16* __restrict__ kn,
    const void* __restrict__ cos_ptr,
    const void* __restrict__ sin_ptr,
    float eps,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int S,
    int H
) {
    // 4 warps per block (128 threads), 1 warp per (s, h, part)
    const int warp_id = ((int)blockIdx.x * 4) + ((int)threadIdx.x >> 5);
    const int lane = (int)threadIdx.x & 31;  // 0..31 -> handles (lane, lane + 32)
    const int total_warps = S * H * 3;
    if (warp_id >= total_warps) return;

    const int part = warp_id % 3;
    const int tmp = warp_id / 3;
    const int h = tmp % H;
    const int s = tmp / H;

    const int64_t in_base = ((int64_t)s * 3 + part) * (int64_t)(H * 64) + (int64_t)h * 64;
    const int64_t out_base = OUT_SHD
        ? (((int64_t)s * H + h) * 64)
        : (((int64_t)h * S + s) * 64);

    const __nv_bfloat16 b0 = qkv[in_base + lane];
    const __nv_bfloat16 b1 = qkv[in_base + lane + 32];

    if (part == 2) {
        v_out[out_base + lane] = b0;
        v_out[out_base + lane + 32] = b1;
        return;
    }

    const float x0 = __bfloat162float(b0);
    const float x1 = __bfloat162float(b1);

    // Warp-level sum of squares across all 64 dims (exact IEEE sqrtf, no --use_fast_math)
    float ss = fmaf(x0, x0, x1 * x1);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, off);
    }
    const float inv_rms = 1.0f / sqrtf(ss * (1.0f / 64.0f) + eps);

    const __nv_bfloat16* wn = (part == 0) ? qn : kn;
    const float h0 = x0 * inv_rms * __bfloat162float(wn[lane]);
    const float h1 = x1 * inv_rms * __bfloat162float(wn[lane + 32]);

    float c0, c1, sn0, sn1;
    const int64_t cs_base = (int64_t)s * 64;
    if constexpr (COS_BF16) {
        const __nv_bfloat16* cp = (const __nv_bfloat16*)cos_ptr;
        const __nv_bfloat16* sp = (const __nv_bfloat16*)sin_ptr;
        c0 = __bfloat162float(cp[cs_base + lane]);
        c1 = __bfloat162float(cp[cs_base + lane + 32]);
        sn0 = __bfloat162float(sp[cs_base + lane]);
        sn1 = __bfloat162float(sp[cs_base + lane + 32]);
    } else {
        const float* cp = (const float*)cos_ptr;
        const float* sp = (const float*)sin_ptr;
        c0 = cp[cs_base + lane];
        c1 = cp[cs_base + lane + 32];
        sn0 = sp[cs_base + lane];
        sn1 = sp[cs_base + lane + 32];
    }

    // NeoX half-rotation:
    //   out[d]    = h[d]    * cos[d]    - h[d+32] * sin[d]
    //   out[d+32] = h[d+32] * cos[d+32] + h[d]    * sin[d+32]
    const float r0 = fmaf(h0, c0, -h1 * sn0);
    const float r1 = fmaf(h1, c1,  h0 * sn1);

    __nv_bfloat16* dst = (part == 0) ? q_out : k_out;
    dst[out_base + lane] = __float2bfloat16_rn(r0);
    dst[out_base + lane + 32] = __float2bfloat16_rn(r1);
}

// ---------------------------------------------------------------------------------------------
// 3. gk_bf16_argmax_cs: Fused BF16 Argmax over [M, V_core] + Core-ID Remap +
//    Mean-Centered Cauchy-Schwarz Bound Check (eliminates 751 MB FP32 logits cast!).
//
//    For each row m in 0..M-1:
//      (best_val, best_local) = argmax_j sub_logits[m, j]
//      out_id[m] = core_ids[best_local]
//      out_val[m] = best_val
//      if check_cs:
//        dot_mu = dot(hidden[m], mu_rest)
//        norm_x = ||hidden[m]||_2
//        out_rescue[m] = (best_val < dot_mu + norm_x * w_rest_bound) ? 1 : 0
//
//    Vectorized uint4 (8 x bf16 = 16 bytes) loads achieve >750 GB/s on L40S.
//    Requires V_core % 8 == 0 and hidden_dim % 2 == 0.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
gk_bf16_argmax_cs_kernel(
    const __nv_bfloat16* __restrict__ sub_logits,  // [M, V_stride]
    const int32_t* __restrict__ core_ids,          // [V_valid] (or nullptr -> identity)
    const __nv_bfloat16* __restrict__ hidden,      // [M, H] (nullable if !check_cs)
    const __nv_bfloat16* __restrict__ mu_rest,     // [H]    (nullable -> zero mean)
    float w_rest_bound,
    int32_t* __restrict__ out_id,                  // [M]
    float* __restrict__ out_val,                   // [M] (nullable)
    uint8_t* __restrict__ out_rescue,              // [M] (nullable if !check_cs)
    int M,
    int V_stride,
    int V_valid,
    int H,
    int check_cs
) {
    const int m = (int)blockIdx.x;
    if (m >= M) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // 1) Vectorized argmax across V_valid (8 bf16 per uint4 load when 16B aligned)
    const __nv_bfloat16* row_ptr = sub_logits + (int64_t)m * V_stride;
    const int n_u4 = V_valid >> 3;
    const bool aligned16 = ((((uintptr_t)row_ptr) & 15u) == 0u);
    float best_v = -3.402823466e+38f;
    int best_i = 0;

    if (aligned16) {
        const uint4* row_u4 = (const uint4*)row_ptr;
        for (int idx = tid; idx < n_u4; idx += 256) {
            const uint4 pack = row_u4[idx];
            const __nv_bfloat16* bp = (const __nv_bfloat16*)&pack;
            const int base = idx << 3;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = __bfloat162float(bp[k]);
                if (v > best_v) {
                    best_v = v;
                    best_i = base + k;
                }
            }
        }
        const int tail_start = n_u4 << 3;
        if (tid < (V_valid - tail_start)) {
            const int c = tail_start + tid;
            const float v = __bfloat162float(row_ptr[c]);
            if (v > best_v || (v == best_v && c < best_i)) {
                best_v = v;
                best_i = c;
            }
        }
    } else {
        for (int c = tid; c < V_valid; c += 256) {
            const float v = __bfloat162float(row_ptr[c]);
            if (v > best_v) {
                best_v = v;
                best_i = c;
            }
        }
    }

    // Warp argmax reduction
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    if (lane == 0) {
        s_val[warp] = best_v;
        s_idx[warp] = best_i;
    }

    // 2) Simultaneously compute ||x||^2 and dot(x, mu_rest) if check_cs
    float local_xx = 0.0f;
    float local_xm = 0.0f;
    if (check_cs && hidden != nullptr) {
        const __nv_bfloat16* xr = hidden + (int64_t)m * H;
        if (mu_rest != nullptr) {
            for (int d = tid; d < H; d += 256) {
                const float xv = __bfloat162float(xr[d]);
                const float mv = __bfloat162float(mu_rest[d]);
                local_xx = fmaf(xv, xv, local_xx);
                local_xm = fmaf(xv, mv, local_xm);
            }
        } else {
            for (int d = tid; d < H; d += 256) {
                const float xv = __bfloat162float(xr[d]);
                local_xx = fmaf(xv, xv, local_xx);
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_xx += __shfl_xor_sync(0xffffffffu, local_xx, off);
            local_xm += __shfl_xor_sync(0xffffffffu, local_xm, off);
        }
    }
    __shared__ float s_xx[8];
    __shared__ float s_xm[8];
    if (check_cs && lane == 0) {
        s_xx[warp] = local_xx;
        s_xm[warp] = local_xm;
    }

    __syncthreads();

    if (tid == 0) {
        float bv = s_val[0];
        int bi = s_idx[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_idx[w] < bi)) {
                bv = s_val[w];
                bi = s_idx[w];
            }
        }
        out_id[m] = (core_ids != nullptr) ? core_ids[bi] : bi;
        if (out_val != nullptr) out_val[m] = bv;
        if (check_cs && out_rescue != nullptr) {
            float xx = 0.0f, xm = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                xx += s_xx[w];
                xm += s_xm[w];
            }
            const float upper = fmaf(sqrtf(xx), w_rest_bound, xm);
            out_rescue[m] = (bv < upper) ? (uint8_t)1 : (uint8_t)0;
        }
    }
}

// =============================================================================================
// v6 fused kernels for the W8A8 FP8 path (per-channel weights, per-token dynamic activations).
// Every kernel reproduces the arithmetic of the kernels it replaces, so its outputs are bitwise
// equal to the unfused SGLang path (checked by tools/test_fused_kernels.py on the box).
// =============================================================================================
__device__ __forceinline__ void t2_unpack8(const uint4 v, float f[8]) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 t2_pack8(const float f[8]) {
    uint4 v;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
}

__device__ __forceinline__ uint4 t2_shfl_xor_u4(const uint4 a, int m) {
    uint4 b;
    b.x = __shfl_xor_sync(0xffffffffu, a.x, m);
    b.y = __shfl_xor_sync(0xffffffffu, a.y, m);
    b.z = __shfl_xor_sync(0xffffffffu, a.z, m);
    b.w = __shfl_xor_sync(0xffffffffu, a.w, m);
    return b;
}

// ---------------------------------------------------------------------------------------------
// 4. gk_vis_qk_norm_rope_inplace: K2 without the split copies.
//    qkv [S, 3, H, 64] bf16 (the qkv_proj output) is updated IN PLACE: Q and K heads get the
//    head RMSNorm + NeoX 2D RoPE, V is not touched. The caller hands FA2-varlen / SDPA strided
//    views of qkv (last dim contiguous), so the [S, H, 64] q / k / v copies of K2 disappear.
//    Thread layout: 8 threads per (token, part, head), thread t owns dims [8t, 8t + 8) as one
//    16-byte vector; its NeoX partner dims (+-32) live in thread t ^ 4 (one shuffle).
//    Bitwise equal to gk_vis_qkv_sdpa_bf16_kernel: the sum of squares replays K2's butterfly
//    (old lane d = 8 (t & 3) + e: xor 16 -> thread t ^ 2, xor 8 -> t ^ 1, xor 4 / 2 / 1 inside
//    the thread) and every product / fma is the same expression.
// ---------------------------------------------------------------------------------------------
template <bool COS_BF16>
__global__ void __launch_bounds__(256)
gk_vis_qk_norm_rope_inplace_kernel(
    __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qn,
    const __nv_bfloat16* __restrict__ kn,
    const void* __restrict__ cos_ptr,
    const void* __restrict__ sin_ptr,
    float eps,
    int S,
    int H
) {
    const int per_tok = H * 16;                 // 2 parts x H heads x 8 threads
    const int64_t total = (int64_t)S * per_tok; // multiple of 32 for even H: warp-uniform trips
    const int64_t gstride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gstride) {
        const int s = (int)(i / per_tok);
        const int r = (int)(i - (int64_t)s * per_tok);
        const int part = r / (H * 8);           // 0 = Q, 1 = K
        const int h = (r >> 3) % H;
        const int t = r & 7;
        uint4* p = reinterpret_cast<uint4*>(qkv + ((int64_t)s * 3 + part) * (int64_t)(H * 64) + (int64_t)h * 64) + t;
        const uint4 a = *p;
        const uint4 b = t2_shfl_xor_u4(a, 4);
        float x[8], y[8];
        t2_unpack8(a, x);
        t2_unpack8(b, y);
        const bool lo = t < 4;
        float pp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) pp[e] = lo ? fmaf(x[e], x[e], y[e] * y[e]) : fmaf(y[e], y[e], x[e] * x[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) pp[e] += __shfl_xor_sync(0xffffffffu, pp[e], 2);
#pragma unroll
        for (int e = 0; e < 8; ++e) pp[e] += __shfl_xor_sync(0xffffffffu, pp[e], 1);
        const float ss = ((pp[0] + pp[4]) + (pp[2] + pp[6])) + ((pp[1] + pp[5]) + (pp[3] + pp[7]));
        const float inv_rms = 1.0f / sqrtf(ss * (1.0f / 64.0f) + eps);

        const __nv_bfloat16* wn = (part == 0) ? qn : kn;
        float wo[8], wp[8];
        t2_unpack8(*reinterpret_cast<const uint4*>(wn + 8 * t), wo);
        t2_unpack8(*reinterpret_cast<const uint4*>(wn + 8 * (t ^ 4)), wp);

        float c[8], sn[8];
        const int64_t cs = (int64_t)s * 64 + 8 * t;
        if constexpr (COS_BF16) {
            t2_unpack8(*reinterpret_cast<const uint4*>((const __nv_bfloat16*)cos_ptr + cs), c);
            t2_unpack8(*reinterpret_cast<const uint4*>((const __nv_bfloat16*)sin_ptr + cs), sn);
        } else {
            const float4* cp = reinterpret_cast<const float4*>((const float*)cos_ptr + cs);
            const float4* sp = reinterpret_cast<const float4*>((const float*)sin_ptr + cs);
            const float4 c0 = cp[0], c1 = cp[1], s0 = sp[0], s1 = sp[1];
            c[0] = c0.x; c[1] = c0.y; c[2] = c0.z; c[3] = c0.w; c[4] = c1.x; c[5] = c1.y; c[6] = c1.z; c[7] = c1.w;
            sn[0] = s0.x; sn[1] = s0.y; sn[2] = s0.z; sn[3] = s0.w; sn[4] = s1.x; sn[5] = s1.y; sn[6] = s1.z; sn[7] = s1.w;
        }
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float ho = x[e] * inv_rms * wo[e];
            const float hp = y[e] * inv_rms * wp[e];
            // K2: out[d] = fmaf(h[d], cos[d], -h[d+32] * sin[d]); out[d+32] = fmaf(h[d+32], cos[d+32], h[d] * sin[d+32])
            o[e] = lo ? fmaf(ho, c[e], -hp * sn[e]) : fmaf(ho, c[e], hp * sn[e]);
        }
        *p = t2_pack8(o);
    }
}

// ---------------------------------------------------------------------------------------------
// 5. gk_silu_mul_quant_fp8: y = bf16(silu(gate) * up) (flashinfer act_and_mul with sgl-kernel's
//    silu<float>), then sgl-kernel's per-token FP8 quant of y (scale = max|y| / 448, e4m3 with
//    saturation), in one pass: gate_up is read once, y never touches memory.
//    One 128-thread CTA per row; each thread keeps up to MAXV 8-element vectors of y in registers.
//    ACT selects the silu arithmetic, QNT the scale arithmetic (both picked by the bitwise test).
// ---------------------------------------------------------------------------------------------
template <int ACT>
__device__ __forceinline__ float t2_silu(float v) {
    if constexpr (ACT == 0) {
        return v / (1.0f + expf(-v));
    } else {
        return __fdividef(v, 1.0f + __expf(-v));
    }
}

__device__ __forceinline__ float t2_block_max128(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    return fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
}

template <int QNT>
__device__ __forceinline__ void t2_fp8_scale(float amax, float& s, float& sinv) {
    if constexpr (QNT == 0) {
        s = amax / 448.0f;
        sinv = (s == 0.0f) ? 0.0f : 1.0f / s;
    } else if constexpr (QNT == 1) {
        s = amax / 448.0f;
        sinv = (amax == 0.0f) ? 0.0f : 448.0f / amax;
    } else if constexpr (QNT == 3) {
        // Exact power-of-2 shift (448 / 16 = 28.0f): keeps 10 full octaves of normal e4m3fn with zero mantissa loss,
        // preventing FP16 accumulator overflow in sm_89 733-TFLOPS mma.f16.e4m3.e4m3.f16 GEMMs.
        s = amax / 28.0f;
        sinv = (s == 0.0f) ? 0.0f : 1.0f / s;
    } else {
        s = amax * (1.0f / 448.0f);
        sinv = (s == 0.0f) ? 0.0f : 1.0f / s;
    }
}

__device__ __forceinline__ uint2 t2_quant8(const float f[8], float sinv) {
    uint2 out;
    uint8_t* b = reinterpret_cast<uint8_t*>(&out);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float v = fmaxf(fminf(f[e] * sinv, 448.0f), -448.0f);
        b[e] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
    return out;
}

template <int MAXV, int ACT, int QNT>
__global__ void __launch_bounds__(128)
gk_silu_mul_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ gu,
    int64_t gu_stride,
    uint8_t* __restrict__ q,
    float* __restrict__ scale,
    int I
) {
    __shared__ float red[4];
    const int m = blockIdx.x;
    const __nv_bfloat16* g = gu + (int64_t)m * gu_stride;
    const __nv_bfloat16* u = g + I;
    const int nv = I >> 3;
    uint4 yv[MAXV];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int v = threadIdx.x + k * 128;
        if (v < nv) {
            float gf[8], uf[8], r[8];
            t2_unpack8(reinterpret_cast<const uint4*>(g)[v], gf);
            t2_unpack8(reinterpret_cast<const uint4*>(u)[v], uf);
#pragma unroll
            for (int e = 0; e < 8; ++e) r[e] = t2_silu<ACT>(gf[e]) * uf[e];
            yv[k] = t2_pack8(r);
            float rb[8];
            t2_unpack8(yv[k], rb);
#pragma unroll
            for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(rb[e]));
        }
    }
    amax = t2_block_max128(amax, red);
    float s, sinv;
    t2_fp8_scale<QNT>(amax, s, sinv);
    if (threadIdx.x == 0) scale[m] = s;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int v = threadIdx.x + k * 128;
        if (v < nv) {
            float rb[8];
            t2_unpack8(yv[k], rb);
            reinterpret_cast<uint2*>(q + (int64_t)m * I)[v] = t2_quant8(rb, sinv);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 6. gk_quant_fp8_rows: the per-token FP8 quant alone (same arithmetic as 5), for the test /
//    for callers that already hold the bf16 activations.
// ---------------------------------------------------------------------------------------------
template <int MAXV, int QNT>
__global__ void __launch_bounds__(128)
gk_quant_fp8_rows_kernel(const __nv_bfloat16* __restrict__ x, int64_t x_stride, uint8_t* __restrict__ q,
                         float* __restrict__ scale, int K) {
    __shared__ float red[4];
    const int m = blockIdx.x;
    const __nv_bfloat16* row = x + (int64_t)m * x_stride;
    const int nv = K >> 3;
    uint4 xv[MAXV];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int v = threadIdx.x + k * 128;
        if (v < nv) {
            xv[k] = reinterpret_cast<const uint4*>(row)[v];
            float f[8];
            t2_unpack8(xv[k], f);
#pragma unroll
            for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
        }
    }
    amax = t2_block_max128(amax, red);
    float s, sinv;
    t2_fp8_scale<QNT>(amax, s, sinv);
    if (threadIdx.x == 0) scale[m] = s;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int v = threadIdx.x + k * 128;
        if (v < nv) {
            float f[8];
            t2_unpack8(xv[k], f);
            reinterpret_cast<uint2*>(q + (int64_t)m * K)[v] = t2_quant8(f, sinv);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 7. gk_norm_quant_fp8: RMSNorm (+ residual add) -> per-token FP8, in one pass.
//    Norm arithmetic = flashinfer 0.6.7's CuTe-DSL RMSNormKernel / FusedAddRMSNormKernel for H = 256 * NB
//    (32 threads per row; thread t owns the 8-wide vectors at columns 8 t + 256 b, b < NB; per-thread
//    sum of squares (order: RED), butterfly shuffles with offsets 1, 2, 4, 8, 16, mean = sum / H (DIV),
//    rstd = rsqrt.approx.ftz(mean + eps), y = bf16((h * rstd) * (w + 0))). Quant = sgl-kernel's
//    per-token FP8 (scale = max|y| / 448) on the bf16 y, i.e. the same as t2_quant8 above.
//    MODE 0: h = x                                         (plain RMSNorm)
//    MODE 1: h = x + r (fp32), r <- bf16(h)                (FusedAddRMSNorm: r = residual, updated in place)
//    MODE 2: z = bf16(RMSNorm(x; w0)), h = z + r, r <- bf16(h)  (GLM: post_self_attn_layernorm, then
//            post_attention_layernorm's fused add)
//    MODE 3: h = bf16(x + r) (torch add), r <- h            (ViT: x_after_add + mlp_out, then norm1)
//    Optional y_out keeps the bf16 normalized row (testing / callers that need it).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float t2_rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

template <int NB, int RED>
__device__ __forceinline__ float t2_row_sumsq(const float (&h)[NB][8]) {
    float s = 0.0f;
    if constexpr (RED == 0) {            // ordered over (v fastest, then b); separate multiply / add
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
            for (int v = 0; v < 8; ++v) s = __fadd_rn(s, __fmul_rn(h[b][v], h[b][v]));
    } else if constexpr (RED == 1) {     // ordered, fused multiply-add
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
            for (int v = 0; v < 8; ++v) s = __fmaf_rn(h[b][v], h[b][v], s);
    } else if constexpr (RED == 2) {     // ordered over (b fastest, then v)
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int b = 0; b < NB; ++b) s = __fadd_rn(s, __fmul_rn(h[b][v], h[b][v]));
    } else {                             // RED 3: ordered over (b fastest, then v), fused
#pragma unroll
        for (int v = 0; v < 8; ++v)
#pragma unroll
            for (int b = 0; b < NB; ++b) s = __fmaf_rn(h[b][v], h[b][v], s);
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
    return s;
}

template <int NB, int RED, int DIV>
__device__ __forceinline__ float t2_rstd(const float (&h)[NB][8], float eps) {
    const float ss = t2_row_sumsq<NB, RED>(h);
    const float mean = DIV == 0 ? __fdiv_rn(ss, (float)(NB * 256)) : __fdividef(ss, (float)(NB * 256));
    return t2_rsqrt_approx(__fadd_rn(mean, eps));
}

template <int NB, int RED, int DIV, int MODE, bool QUANT, int QNT = 0>
__global__ void __launch_bounds__(128)
gk_norm_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_stride,
    __nv_bfloat16* __restrict__ r, int64_t r_stride,
    const __nv_bfloat16* __restrict__ w0, float eps0,
    const __nv_bfloat16* __restrict__ w, float eps,
    __nv_bfloat16* __restrict__ y_out, int64_t y_stride,
    uint8_t* __restrict__ q, float* __restrict__ scale,
    int M
) {
    const int row = (int)blockIdx.x * 4 + ((int)threadIdx.x >> 5);
    if (row >= M) return;                   // whole warps exit together (one row per warp)
    const int t = (int)threadIdx.x & 31;
    const __nv_bfloat16* xr = x + (int64_t)row * x_stride;
    float h[NB][8];
#pragma unroll
    for (int b = 0; b < NB; ++b) t2_unpack8(reinterpret_cast<const uint4*>(xr)[t + 32 * b], h[b]);
    if constexpr (MODE == 2) {              // z = bf16(RMSNorm(x; w0))
        const float rs0 = t2_rstd<NB, RED, DIV>(h, eps0);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            float wv[8], z[8];
            t2_unpack8(reinterpret_cast<const uint4*>(w0)[t + 32 * b], wv);
#pragma unroll
            for (int v = 0; v < 8; ++v) z[v] = __fmul_rn(__fmul_rn(h[b][v], rs0), __fadd_rn(wv[v], 0.0f));
            t2_unpack8(t2_pack8(z), h[b]);  // round to bf16 like the stored RMSNorm output
        }
    }
    if constexpr (MODE >= 1) {
        __nv_bfloat16* rr = r + (int64_t)row * r_stride;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            float rv[8];
            t2_unpack8(reinterpret_cast<const uint4*>(rr)[t + 32 * b], rv);
#pragma unroll
            for (int v = 0; v < 8; ++v) h[b][v] = __fadd_rn(h[b][v], rv[v]);
            const uint4 hb = t2_pack8(h[b]);
            reinterpret_cast<uint4*>(rr)[t + 32 * b] = hb;
            if constexpr (MODE == 3) t2_unpack8(hb, h[b]);   // torch's bf16 add: the norm sees the rounded sum
        }
    }
    const float rs = t2_rstd<NB, RED, DIV>(h, eps);
    uint4 yv[NB];
    float amax = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float wv[8], y[8];
        t2_unpack8(reinterpret_cast<const uint4*>(w)[t + 32 * b], wv);
#pragma unroll
        for (int v = 0; v < 8; ++v) y[v] = __fmul_rn(__fmul_rn(h[b][v], rs), __fadd_rn(wv[v], 0.0f));
        yv[b] = t2_pack8(y);
        if (y_out != nullptr) reinterpret_cast<uint4*>(y_out + (int64_t)row * y_stride)[t + 32 * b] = yv[b];
        if constexpr (QUANT) {
            float yb[8];
            t2_unpack8(yv[b], yb);
#pragma unroll
            for (int v = 0; v < 8; ++v) amax = fmaxf(amax, fabsf(yb[v]));
        }
    }
    if constexpr (QUANT) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        float s, sinv;
        t2_fp8_scale<QNT>(amax, s, sinv);
        if (t == 0) scale[row] = s;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            float yb[8];
            t2_unpack8(yv[b], yb);
            reinterpret_cast<uint2*>(q + (int64_t)row * (NB * 256))[t + 32 * b] = t2_quant8(yb, sinv);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// C-ABI entrypoints called via ctypes on PyTorch's active cudaStream_t (CUDA Graph safe!)
// ---------------------------------------------------------------------------------------------
extern "C" {

// v6: ABI version of the fused kernels (bump on any signature change)
int track2_fused_abi(void) { return 1; }

int track2_vis_qk_norm_rope_inplace(
    void* qkv_bf16, const void* qn_bf16, const void* kn_bf16, const void* cos_ptr, const void* sin_ptr,
    int cos_is_bf16, float eps, int S, int H, void* stream_ptr
) {
    if (S <= 0 || H <= 0) return 0;
    if (H & 1) return 1001;  // warp-uniform shuffles need an even head count
    const int64_t total = (int64_t)S * H * 16;
    int64_t blocks = (total + 255) / 256;
    if (blocks > 142 * 12) blocks = 142 * 12;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (cos_is_bf16) {
        gk_vis_qk_norm_rope_inplace_kernel<true><<<(int)blocks, 256, 0, stream>>>(
            (__nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16, cos_ptr, sin_ptr, eps, S, H);
    } else {
        gk_vis_qk_norm_rope_inplace_kernel<false><<<(int)blocks, 256, 0, stream>>>(
            (__nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16, cos_ptr, sin_ptr, eps, S, H);
    }
    return (int)cudaGetLastError();
}

#define T2_SMQ_CASE(MV, A, Q)                                                                          \
    gk_silu_mul_quant_fp8_kernel<MV, A, Q><<<M, 128, 0, stream>>>((const __nv_bfloat16*)gate_up_bf16,  \
                                                                  gu_stride, (uint8_t*)q_e4m3, (float*)scale_f32, I)

int track2_silu_mul_quant_fp8(
    const void* gate_up_bf16, int64_t gu_stride, void* q_e4m3, void* scale_f32, int M, int I, int act_mode,
    int qnt_mode, void* stream_ptr
) {
    if (M <= 0) return 0;
    if (I <= 0 || (I & 7) || I > 128 * 8 * 6) return 1002;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int nv = I >> 3;
    const int mv = (nv + 127) / 128;
    const int key = act_mode * 10 + qnt_mode;
    if (mv <= 4) {
        switch (key) {
            case 0: T2_SMQ_CASE(4, 0, 0); break;
            case 1: T2_SMQ_CASE(4, 0, 1); break;
            case 2: T2_SMQ_CASE(4, 0, 2); break;
            case 3: T2_SMQ_CASE(4, 0, 3); break;
            case 10: T2_SMQ_CASE(4, 1, 0); break;
            case 11: T2_SMQ_CASE(4, 1, 1); break;
            case 12: T2_SMQ_CASE(4, 1, 2); break;
            case 13: T2_SMQ_CASE(4, 1, 3); break;
            default: return 1003;
        }
    } else {
        switch (key) {
            case 0: T2_SMQ_CASE(6, 0, 0); break;
            case 1: T2_SMQ_CASE(6, 0, 1); break;
            case 2: T2_SMQ_CASE(6, 0, 2); break;
            case 3: T2_SMQ_CASE(6, 0, 3); break;
            case 10: T2_SMQ_CASE(6, 1, 0); break;
            case 11: T2_SMQ_CASE(6, 1, 1); break;
            case 12: T2_SMQ_CASE(6, 1, 2); break;
            case 13: T2_SMQ_CASE(6, 1, 3); break;
            default: return 1003;
        }
    }
    return (int)cudaGetLastError();
}

#define T2_NQ_LAUNCH(NB_, RED_, DIV_, MODE_, QNT_)                                                                \
    gk_norm_quant_fp8_kernel<NB_, RED_, DIV_, MODE_, true, QNT_><<<blocks, 128, 0, stream>>>(                     \
        (const __nv_bfloat16*)x_bf16, x_stride, (__nv_bfloat16*)r_bf16, r_stride, (const __nv_bfloat16*)w0_bf16, eps0, \
        (const __nv_bfloat16*)w_bf16, eps, (__nv_bfloat16*)y_bf16, y_stride, (uint8_t*)q_e4m3, (float*)scale_f32, M)

#define T2_NQ_MODES(NB_, RED_, DIV_, QNT_)                         \
    switch (mode) {                                                \
        case 0: T2_NQ_LAUNCH(NB_, RED_, DIV_, 0, QNT_); break;     \
        case 1: T2_NQ_LAUNCH(NB_, RED_, DIV_, 1, QNT_); break;     \
        case 2: T2_NQ_LAUNCH(NB_, RED_, DIV_, 2, QNT_); break;     \
        case 3: T2_NQ_LAUNCH(NB_, RED_, DIV_, 3, QNT_); break;     \
        default: return 1004;                                      \
    }

#define T2_NQ_RED(NB_, DIV_, QNT_)                                 \
    switch (red) {                                                 \
        case 0: T2_NQ_MODES(NB_, 0, DIV_, QNT_); break;            \
        case 1: T2_NQ_MODES(NB_, 1, DIV_, QNT_); break;            \
        case 2: T2_NQ_MODES(NB_, 2, DIV_, QNT_); break;            \
        case 3: T2_NQ_MODES(NB_, 3, DIV_, QNT_); break;            \
        default: return 1005;                                      \
    }

// v6: (residual add +) RMSNorm -> per-token FP8 (gk_norm_quant_fp8_kernel). H must be 1024 or 1536.
// div bit 0: division mode (0=rn, 1=fast); div bit 1: QNT=3 (a_max=28.0f for f16-accum FP8 GEMM).
int track2_norm_quant_fp8(
    const void* x_bf16, int64_t x_stride, void* r_bf16, int64_t r_stride, const void* w0_bf16, float eps0,
    const void* w_bf16, float eps, void* y_bf16, int64_t y_stride, void* q_e4m3, void* scale_f32, int M, int H,
    int mode, int red, int div, void* stream_ptr
) {
    if (M <= 0) return 0;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int blocks = (M + 3) / 4;
    const int div_mode = div & 1;
    const int qnt_f16 = (div & 2) != 0;
    if (H == 1536) {
        if (qnt_f16) {
            if (div_mode == 0) { T2_NQ_RED(6, 0, 3); } else { T2_NQ_RED(6, 1, 3); }
        } else {
            if (div_mode == 0) { T2_NQ_RED(6, 0, 0); } else { T2_NQ_RED(6, 1, 0); }
        }
    } else if (H == 1024) {
        if (qnt_f16) {
            if (div_mode == 0) { T2_NQ_RED(4, 0, 3); } else { T2_NQ_RED(4, 1, 3); }
        } else {
            if (div_mode == 0) { T2_NQ_RED(4, 0, 0); } else { T2_NQ_RED(4, 1, 0); }
        }
    } else {
        return 1006;
    }
    return (int)cudaGetLastError();
}

int track2_quant_fp8_rows(
    const void* x_bf16, int64_t x_stride, void* q_e4m3, void* scale_f32, int M, int K, int qnt_mode, void* stream_ptr
) {
    if (M <= 0) return 0;
    if (K <= 0 || (K & 7) || K > 128 * 8 * 6) return 1002;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int mv = ((K >> 3) + 127) / 128;
    const __nv_bfloat16* x = (const __nv_bfloat16*)x_bf16;
    uint8_t* q = (uint8_t*)q_e4m3;
    float* s = (float*)scale_f32;
    if (mv <= 4) {
        if (qnt_mode == 0) gk_quant_fp8_rows_kernel<4, 0><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else if (qnt_mode == 1) gk_quant_fp8_rows_kernel<4, 1><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else if (qnt_mode == 3) gk_quant_fp8_rows_kernel<4, 3><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else gk_quant_fp8_rows_kernel<4, 2><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
    } else {
        if (qnt_mode == 0) gk_quant_fp8_rows_kernel<6, 0><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else if (qnt_mode == 1) gk_quant_fp8_rows_kernel<6, 1><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else if (qnt_mode == 3) gk_quant_fp8_rows_kernel<6, 3><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
        else gk_quant_fp8_rows_kernel<6, 2><<<M, 128, 0, stream>>>(x, x_stride, q, s, K);
    }
    return (int)cudaGetLastError();
}

int track2_u8_norm_bf16(
    const void* src_u8,
    void* dst_bf16,
    int64_t n_patches,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2,
    void* stream_ptr
) {
    if (n_patches <= 0) return 0;
    const int64_t total = n_patches * 588;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    gk_u8_norm_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)src_u8, (__nv_bfloat16*)dst_bf16, n_patches,
        s0, b0, s1, b1, s2, b2
    );
    return (int)cudaGetLastError();
}

int track2_raw_rgb_patchify_norm_bf16(
    const void* rgb_hwc_u8,
    void* dst_patches_bf16,
    int gh,
    int gw,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2,
    void* stream_ptr
) {
    const int n_patches = gh * gw;
    if (n_patches <= 0) return 0;
    const int64_t total = (int64_t)n_patches * 588;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    gk_raw_rgb_patchify_norm_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)rgb_hwc_u8, (__nv_bfloat16*)dst_patches_bf16,
        gh, gw, s0, b0, s1, b1, s2, b2
    );
    return (int)cudaGetLastError();
}

int track2_chw_u8_patchify_norm_bf16(
    const void* rgb_chw_u8,
    void* dst_patches_bf16,
    int gh,
    int gw,
    float s0, float b0,
    float s1, float b1,
    float s2, float b2,
    void* stream_ptr
) {
    const int n_patches = gh * gw;
    if (n_patches <= 0) return 0;
    const int64_t total = (int64_t)n_patches * 588;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    gk_chw_u8_patchify_norm_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)rgb_chw_u8, (__nv_bfloat16*)dst_patches_bf16,
        gh, gw, s0, b0, s1, b1, s2, b2
    );
    return (int)cudaGetLastError();
}

int track2_vis_qkv_sdpa_bf16(
    const void* qkv_bf16,
    const void* qn_bf16,
    const void* kn_bf16,
    const void* cos_ptr,
    const void* sin_ptr,
    int cos_is_bf16,
    int out_shd,
    float eps,
    void* q_out_bf16,
    void* k_out_bf16,
    void* v_out_bf16,
    int S,
    int H,
    void* stream_ptr
) {
    if (S <= 0 || H <= 0) return 0;
    const int total_warps = S * H * 3;
    const int blocks = (total_warps + 3) / 4;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (cos_is_bf16) {
        if (out_shd) {
            gk_vis_qkv_sdpa_bf16_kernel<true, true><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16,
                cos_ptr, sin_ptr, eps,
                (__nv_bfloat16*)q_out_bf16, (__nv_bfloat16*)k_out_bf16, (__nv_bfloat16*)v_out_bf16, S, H
            );
        } else {
            gk_vis_qkv_sdpa_bf16_kernel<true, false><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16,
                cos_ptr, sin_ptr, eps,
                (__nv_bfloat16*)q_out_bf16, (__nv_bfloat16*)k_out_bf16, (__nv_bfloat16*)v_out_bf16, S, H
            );
        }
    } else {
        if (out_shd) {
            gk_vis_qkv_sdpa_bf16_kernel<false, true><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16,
                cos_ptr, sin_ptr, eps,
                (__nv_bfloat16*)q_out_bf16, (__nv_bfloat16*)k_out_bf16, (__nv_bfloat16*)v_out_bf16, S, H
            );
        } else {
            gk_vis_qkv_sdpa_bf16_kernel<false, false><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)qkv_bf16, (const __nv_bfloat16*)qn_bf16, (const __nv_bfloat16*)kn_bf16,
                cos_ptr, sin_ptr, eps,
                (__nv_bfloat16*)q_out_bf16, (__nv_bfloat16*)k_out_bf16, (__nv_bfloat16*)v_out_bf16, S, H
            );
        }
    }
    return (int)cudaGetLastError();
}

int track2_bf16_argmax_cs(
    const void* sub_logits_bf16,
    const void* core_ids_i32,
    const void* hidden_bf16,
    const void* mu_rest_bf16,
    float w_rest_bound,
    void* out_id_i32,
    void* out_val_f32,
    void* out_rescue_u8,
    int M,
    int V_stride,
    int V_valid,
    int H,
    int check_cs,
    void* stream_ptr
) {
    if (M <= 0 || V_stride <= 0 || V_valid <= 0) return 0;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    gk_bf16_argmax_cs_kernel<<<M, 256, 0, stream>>>(
        (const __nv_bfloat16*)sub_logits_bf16,
        (const int32_t*)core_ids_i32,
        (const __nv_bfloat16*)hidden_bf16,
        (const __nv_bfloat16*)mu_rest_bf16,
        w_rest_bound,
        (int32_t*)out_id_i32,
        (float*)out_val_f32,
        (uint8_t*)out_rescue_u8,
        M, V_stride, V_valid, H, check_cs
    );
    return (int)cudaGetLastError();
}

}  // extern "C"
