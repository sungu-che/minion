// Track 2 (GLM-OCR) Native BF16 CUDA (sm_89 / Ada Lovelace L40S) Fused Kernels.
// Upgraded from rust_gpu/kernels/gravity.hip to native __nv_bfloat16 + warp-shuffle intrinsics
// for zero-copy PyTorch data_ptr() integration on torch.cuda.current_stream() (100% CUDA Graph safe).
//
// Build inside track2-base:v1 (no --use_fast_math to preserve exact IEEE sqrtf/RoPE precision):
//   nvcc -O3 -std=c++17 -arch=sm_89 --shared -Xcompiler -fPIC \
//        rust_gpu/kernels/track2_cuda_sm89.cu -o gravity_ocr/_native/libtrack2_cuda_sm89.so

#include <cuda_bf16.h>
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

// ---------------------------------------------------------------------------------------------
// C-ABI entrypoints called via ctypes on PyTorch's active cudaStream_t (CUDA Graph safe!)
// ---------------------------------------------------------------------------------------------
extern "C" {

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
