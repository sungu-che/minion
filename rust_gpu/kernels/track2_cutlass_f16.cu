// Track 2 Ada sm_89 733-TFLOPS FP8xFP8->FP16-Accumulator EVT GEMM (CUTLASS 4.4.2).
//
// Emits native QMMA.16832.F16.E4M3.E4M3 tensor-core instructions (2x throughput of FP32-accum)
// with fused EVT epilogue: D_bf16 = bf16((fp32(acc_f16) * scale_b[n]) * scale_a[m] + bias[n]).
//
// Requires power-of-2 scaled inputs (QNT=3: a_max=28.0) and weights (shift=32: w_max=14.0) so
// FP16 accumulation never overflows even at K=4608 (rel error vs FP32-accum CUTLASS: 0.00066-0.00119).
//
// Build inside track2-base:v1 on L40S (sm_89):
//   CUT=/opt/sglang/.venv-sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -Xcompiler -fPIC -shared \
//     -I"$CUT" -I"$CUT/../tools/util/include" \
//     rust_gpu/kernels/track2_cutlass_f16.cu -o gravity_ocr/_native/libtrack2_cutlass_f16.so

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace {

template <int BM, int BN, int BK, int WM, int WN, int WK, int STAGES, bool USE_STREAMK>
struct Sm89Fp8F16Gemm {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementEpilogue = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
  using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int kEpilogueStages = 1;
  static constexpr int kAlignC = 8;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementC, kAlignC, kEpilogueStages>;

  using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>, false>;
  using ScaleA = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::_1, cute::_0, cute::_0>, false>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, cutlass::bfloat16_t, cute::Stride<cute::_0, cute::_1, cute::_0>, true>;

  using ComputeMulB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<ComputeMulB, AccFetch, ScaleB>;

  using ComputeFmaA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiply_add, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<ComputeFmaA, EVT0, ScaleA, Bias>;

  using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementC, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using EVTFusion = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

  using Swizzle = std::conditional_t<
      USE_STREAMK,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, 16,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, 16,
      ElementC, LayoutC, kAlignC,
      ElementAccumulator, ElementEpilogue,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      ThreadblockShape, WarpShape, InstructionShape,
      EVTFusion, Swizzle, STAGES,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static int run(
      const void* A, const void* B, const float* sa, const float* sb,
      const void* bias, void* D, int M, int N, int K,
      void* workspace, size_t ws_bytes, cudaStream_t stream) {
    typename EVTFusion::Arguments callback_args{
      {
        {
          {},
          {sb, float(0), {_0{}, _1{}, _0{}}},
          {}
        },
        {sa, float(0), {_1{}, _0{}, _0{}}},
        {reinterpret_cast<const cutlass::bfloat16_t*>(bias), cutlass::bfloat16_t(0.0f), {_0{}, _1{}, _0{}}},
        {}
      },
      {reinterpret_cast<cutlass::bfloat16_t*>(D), {int64_t(N), _1{}, _0{}}}
    };
    cutlass::gemm::GemmCoord problem(M, N, K);
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        callback_args,
        reinterpret_cast<const ElementA*>(A),
        reinterpret_cast<const ElementB*>(B),
        nullptr, nullptr,
        0, 0, 0, 0,
        K, K, N, N);
    Gemm gemm;
    size_t need = gemm.get_workspace_size(args);
    if (need > ws_bytes) return -2;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -3;
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) return -4;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : -5;
  }
};

}  // namespace

extern "C" {

int track2_cutlass_f16_abi(void) { return 1; }

int track2_cutlass_f16_mm(
    int cfg_id,
    const void* A,
    const void* B,
    const float* sa,
    const float* sb,
    const void* bias,
    void* D,
    int M,
    int N,
    int K,
    void* workspace,
    size_t ws_bytes,
    cudaStream_t stream) {
  switch (cfg_id) {
    case 0: return Sm89Fp8F16Gemm<128, 256,  64, 64, 64, 64, 3, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 1: return Sm89Fp8F16Gemm<128, 128,  64, 64, 64, 64, 4, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 2: return Sm89Fp8F16Gemm<256, 128,  64, 64, 64, 64, 3, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 3: return Sm89Fp8F16Gemm< 64, 128,  64, 32, 64, 64, 5, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 4: return Sm89Fp8F16Gemm< 64, 256,  64, 32, 64, 64, 4, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 5: return Sm89Fp8F16Gemm< 64,  64,  64, 32, 32, 64, 5, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 6: return Sm89Fp8F16Gemm< 64, 128, 128, 32, 64, 64, 3, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 7: return Sm89Fp8F16Gemm< 64, 128,  64, 32, 64, 64, 5,  true>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    case 8: return Sm89Fp8F16Gemm<128, 128, 128, 64, 64, 64, 3, false>::run(A, B, sa, sb, bias, D, M, N, K, workspace, ws_bytes, stream);
    default: return -1;
  }
}

}  // extern "C"
