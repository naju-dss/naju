// Naju v1 scan — warp-shuffle-optimized CUDA kernels.
//
// SAME recurrence and Python-facing interface (fwd / bwd / fwd_chunked) as
// naju_cuda.cu, but re-architected around WARPS instead of block-wide
// shared-memory reductions:
//
//   * one warp handles one (b, d) channel; lane L owns state indices
//     {L, L+32, L+64, ...} (R = ceil(N/32) <= 4 registers per lane for N<=128);
//   * the per-step readout  y = sum_n C_n x_n  is a single 32-lane warp-shuffle
//     reduction — NO shared memory and NO __syncthreads on the hot path
//     (this removes the log(N) syncs/step that dominate the baseline);
//   * the broadcast scalars (u_t, f_t, i_t) are read once via __ldg (L1 bcast);
//   * WPB warps per block raise occupancy on Blackwell's larger SMs.
//
// Numerically identical to the FP32 reference (same op order per state index;
// only the reduction tree changes, associativity within ~1e-6). N <= 128, fp32.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

constexpr int CK   = 16;   // recompute chunk length (training bwd)
constexpr int CKP  = 128;  // chunk length (inference chunk-parallel)
constexpr int WPB  = 8;    // warps per block (occupancy knob)

__device__ __forceinline__ float sigf(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float warpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;  // lane 0 holds the full sum
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ─────────────────────────────── forward ────────────────────────────────────
template <int R>
__global__ void fwd_warp(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    float* __restrict__ y, float* __restrict__ ckpt,
    int B, int T, int D, int N, int nchunks, bool save_ckpt) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bd  = (long)blockIdx.x * (blockDim.x >> 5) + warp;
  if (bd >= (long)B * D) return;
  const int b = bd / D, d = bd % D;
  const float Dval = Dd[d];
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) s[r] = 0.f;

  for (int t = 0; t < T; ++t) {
    const long btd = ((long)b * T + t) * D + d;
    const long bt  = (long)b * T + t;
    if (save_ckpt && (t % CK == 0)) {
#pragma unroll
      for (int r = 0; r < R; ++r) { int n = lane + (r << 5); if (n < N) ckpt[((bd * nchunks) + t / CK) * N + n] = s[r]; }
    }
    const float u_t = __ldg(&u[btd]);
    const float f   = sigf(__ldg(&fl[btd]));
    const float i   = sigf(__ldg(&il[btd]));
    float acc = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = lane + (r << 5);
      const float Bt = (n < N) ? __ldg(&Bm[bt * N + n]) : 0.f;
      const float Ct = (n < N) ? __ldg(&Cm[bt * N + n]) : 0.f;
      s[r] = f * s[r] + i * Bt * u_t;
      acc += Ct * s[r];
    }
    const float yv = warpSum(acc);
    if (lane == 0) y[btd] = yv + Dval * u_t;
  }
}

// ─────────────────────────────── backward ───────────────────────────────────
// Per-warp shared xcur holds the recomputed chunk states [CK][N]; the two per-step
// reductions (d_f, sumdB) are warp shuffles. Shared per block = WPB * CK * N floats.
template <int R>
__global__ void bwd_warp(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    const float* __restrict__ ckpt, const float* __restrict__ dy,
    float* __restrict__ du, float* __restrict__ df, float* __restrict__ di,
    float* __restrict__ dB, float* __restrict__ dC, float* __restrict__ dDacc,
    int B, int T, int D, int N, int nchunks) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bd  = (long)blockIdx.x * (blockDim.x >> 5) + warp;
  extern __shared__ float sh[];                 // [WPB][CK*N]
  float* xcur = sh + (long)warp * CK * N;
  if (bd >= (long)B * D) return;
  const int b = bd / D, d = bd % D;
  const float Dval = Dd[d];
  float carry[R];
#pragma unroll
  for (int r = 0; r < R; ++r) carry[r] = 0.f;
  float dDsum = 0.f;

  for (int c = nchunks - 1; c >= 0; --c) {
    const int t0 = c * CK, len = min(CK, T - t0);
    float s[R], start[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { int n = lane + (r << 5); s[r] = (n < N) ? ckpt[((bd * nchunks) + c) * N + n] : 0.f; start[r] = s[r]; }
    // recompute forward within chunk -> xcur
    for (int tl = 0; tl < len; ++tl) {
      const long btd = ((long)b * T + (t0 + tl)) * D + d;
      const long bt  = (long)b * T + (t0 + tl);
      const float u_t = __ldg(&u[btd]);
      const float f = sigf(__ldg(&fl[btd])), i = sigf(__ldg(&il[btd]));
#pragma unroll
      for (int r = 0; r < R; ++r) { int n = lane + (r << 5); float Bt = (n < N) ? __ldg(&Bm[bt * N + n]) : 0.f; s[r] = f * s[r] + i * Bt * u_t; if (n < N) xcur[tl * N + n] = s[r]; }
    }
    __syncwarp();
    // reverse over chunk
    for (int tl = len - 1; tl >= 0; --tl) {
      const long btd = ((long)b * T + (t0 + tl)) * D + d;
      const long bt  = (long)b * T + (t0 + tl);
      const float u_t = __ldg(&u[btd]);
      const float f = sigf(__ldg(&fl[btd])), i = sigf(__ldg(&il[btd]));
      const float dy_t = __ldg(&dy[btd]);
      float lf = 0.f, lB = 0.f;   // local partials for d_f, sumdB
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int n = lane + (r << 5);
        const float Bt = (n < N) ? __ldg(&Bm[bt * N + n]) : 0.f;
        const float Ct = (n < N) ? __ldg(&Cm[bt * N + n]) : 0.f;
        const float x_t    = (n < N) ? xcur[tl * N + n] : 0.f;
        const float x_prev = (n < N) ? (tl > 0 ? xcur[(tl - 1) * N + n] : start[r]) : 0.f;
        const float d_x = Ct * dy_t + carry[r];
        if (n < N) atomicAdd(&dC[bt * N + n], dy_t * x_t);
        carry[r] = f * d_x;              // -> x_{t-1} for the next reverse step
        lf += d_x * x_prev;              // partial for d_f = sum_n d_x*x_prev
        lB += d_x * Bt;                  // partial for sumdB = sum_n d_x*B
        if (n < N) atomicAdd(&dB[bt * N + n], d_x * i * u_t);
      }
      dDsum += dy_t * u_t;
      const float d_f   = warpSum(lf);
      const float sumdB = warpSum(lB);
      if (lane == 0) {
        df[btd] = d_f * f * (1.f - f);
        di[btd] = (sumdB * u_t) * i * (1.f - i);
        du[btd] = dy_t * Dval + i * sumdB;
      }
    }
  }
  if (lane == 0) dDacc[bd] = dDsum;
}

// ─────────────── chunk-parallel forward (inference) ─────────────────────────
template <int R>
__global__ void p1_warp(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    float* __restrict__ Pf, float* __restrict__ S, int B, int T, int D, int N, int G) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = blockIdx.y;
  const long bd = (long)blockIdx.x * (blockDim.x >> 5) + warp;
  if (bd >= (long)B * D) return;
  const int b = bd / D, d = bd % D;
  const int t0 = c * CKP, len = min(CKP, T - t0);
  float s[R]; float pf = 1.f;
#pragma unroll
  for (int r = 0; r < R; ++r) s[r] = 0.f;
  for (int tl = 0; tl < len; ++tl) {
    const long btd = ((long)b * T + (t0 + tl)) * D + d;
    const long bt  = (long)b * T + (t0 + tl);
    const float f = sigf(__ldg(&fl[btd]));
    const float iu = sigf(__ldg(&il[btd])) * __ldg(&u[btd]);
#pragma unroll
    for (int r = 0; r < R; ++r) { int n = lane + (r << 5); float Bt = (n < N) ? __ldg(&Bm[bt * N + n]) : 0.f; s[r] = f * s[r] + iu * Bt; }
    pf *= f;
  }
#pragma unroll
  for (int r = 0; r < R; ++r) { int n = lane + (r << 5); if (n < N) S[(bd * G + c) * N + n] = s[r]; }
  if (lane == 0) Pf[bd * G + c] = pf;
}

__global__ void p2_seq(
    const float* __restrict__ Pf, const float* __restrict__ S,
    float* __restrict__ Init, int B, int D, int N, int G) {
  const long bd = (long)blockIdx.x * blockDim.y + threadIdx.y;
  const int n = threadIdx.x;
  if (bd >= (long)B * D || n >= N) return;
  float init = 0.f;
  for (int c = 0; c < G; ++c) {
    Init[(bd * G + c) * N + n] = init;
    init = Pf[bd * G + c] * init + S[(bd * G + c) * N + n];
  }
}

template <int R>
__global__ void p3_warp(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    const float* __restrict__ Init, float* __restrict__ y, int B, int T, int D, int N, int G) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int c = blockIdx.y;
  const long bd = (long)blockIdx.x * (blockDim.x >> 5) + warp;
  if (bd >= (long)B * D) return;
  const int b = bd / D, d = bd % D;
  const int t0 = c * CKP, len = min(CKP, T - t0);
  const float Dval = Dd[d];
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { int n = lane + (r << 5); s[r] = (n < N) ? Init[(bd * G + c) * N + n] : 0.f; }
  for (int tl = 0; tl < len; ++tl) {
    const long btd = ((long)b * T + (t0 + tl)) * D + d;
    const long bt  = (long)b * T + (t0 + tl);
    const float u_t = __ldg(&u[btd]);
    const float f = sigf(__ldg(&fl[btd])), iu = sigf(__ldg(&il[btd])) * u_t;
    float acc = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) { int n = lane + (r << 5); float Bt = (n < N) ? __ldg(&Bm[bt * N + n]) : 0.f; float Ct = (n < N) ? __ldg(&Cm[bt * N + n]) : 0.f; s[r] = f * s[r] + iu * Bt; acc += Ct * s[r]; }
    const float yv = warpSum(acc);
    if (lane == 0) y[btd] = yv + Dval * u_t;
  }
}

// ─────────────────────────────── dispatch/launchers ─────────────────────────
// R = ceil(N/32) state indices per lane; dispatched to a templated kernel.
#define DISPATCH_R(R_, ...) do { \
  if      ((R_) == 1) { constexpr int R = 1; __VA_ARGS__; } \
  else if ((R_) == 2) { constexpr int R = 2; __VA_ARGS__; } \
  else if ((R_) == 3) { constexpr int R = 3; __VA_ARGS__; } \
  else                { constexpr int R = 4; __VA_ARGS__; } \
} while (0)

std::vector<torch::Tensor> naju_fwd(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd, bool save_ckpt) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kFloat32, "float32 CUDA required");
  const int B = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  TORCH_CHECK(N <= 128, "d_state <= 128");
  const int nchunks = (T + CK - 1) / CK;
  auto opt = u.options();
  auto y = torch::empty({B, T, D}, opt);
  auto ckpt = save_ckpt ? torch::empty({B, D, nchunks, N}, opt) : torch::empty({0}, opt);
  const long chans = (long)B * D;
  dim3 grid(ceil_div(chans, WPB)); dim3 blk(WPB * 32);
  const int R_ = ceil_div(N, 32);
  DISPATCH_R(R_, fwd_warp<R><<<grid, blk>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      y.data_ptr<float>(), save_ckpt ? ckpt.data_ptr<float>() : nullptr,
      B, T, D, N, nchunks, save_ckpt));
  return {y, ckpt};
}

std::vector<torch::Tensor> naju_bwd(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd,
    torch::Tensor ckpt, torch::Tensor dy) {
  const int B = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  const int nchunks = (T + CK - 1) / CK;
  auto opt = u.options();
  auto du = torch::empty_like(u), df = torch::empty_like(fl), di = torch::empty_like(il);
  auto dB = torch::zeros_like(Bm), dC = torch::zeros_like(Cm);
  auto dDacc = torch::zeros({(long)B * D}, opt);
  const long chans = (long)B * D;
  // xcur = wpb*CK*N floats must fit the 48KB default dynamic-shmem budget; cap wpb.
  const int wpb = std::max(1, std::min((int)WPB, (int)(49152 / (CK * N * sizeof(float)))));
  dim3 grid(ceil_div(chans, wpb)); dim3 blk(wpb * 32);
  const size_t shmem = (size_t)wpb * CK * N * sizeof(float);
  const int R_ = ceil_div(N, 32);
  DISPATCH_R(R_, bwd_warp<R><<<grid, blk, shmem>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      ckpt.data_ptr<float>(), dy.data_ptr<float>(),
      du.data_ptr<float>(), df.data_ptr<float>(), di.data_ptr<float>(),
      dB.data_ptr<float>(), dC.data_ptr<float>(), dDacc.data_ptr<float>(),
      B, T, D, N, nchunks));
  auto dD = dDacc.view({B, D}).sum(0);
  return {du, df, di, dB, dC, dD};
}

torch::Tensor naju_fwd_chunked(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kFloat32, "float32 CUDA required");
  const int B = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  TORCH_CHECK(N <= 128, "d_state <= 128");
  const int G = (T + CKP - 1) / CKP;
  auto opt = u.options();
  auto y = torch::empty({B, T, D}, opt);
  auto Pf = torch::empty({(long)B * D, G}, opt);
  auto S  = torch::empty({(long)B * D, G, N}, opt);
  auto Init = torch::empty({(long)B * D, G, N}, opt);
  const long chans = (long)B * D;
  dim3 grid1(ceil_div(chans, WPB), G); dim3 blk(WPB * 32);
  const int R_ = ceil_div(N, 32);
  DISPATCH_R(R_, p1_warp<R><<<grid1, blk>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Pf.data_ptr<float>(), S.data_ptr<float>(), B, T, D, N, G));
  // p2: block = (N, warps_y); one thread per (n) per channel
  const int wy = 8;
  dim3 grid2(ceil_div(chans, wy)); dim3 blk2(std::max(N, 1), wy);
  p2_seq<<<grid2, blk2>>>(Pf.data_ptr<float>(), S.data_ptr<float>(), Init.data_ptr<float>(), B, D, N, G);
  DISPATCH_R(R_, p3_warp<R><<<grid1, blk>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      Init.data_ptr<float>(), y.data_ptr<float>(), B, T, D, N, G));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &naju_fwd, "Naju v1 forward (warp-opt, sequential + ckpt)");
  m.def("bwd", &naju_bwd, "Naju v1 recompute backward (warp-opt)");
  m.def("fwd_chunked", &naju_fwd_chunked, "Naju v1 chunk-parallel forward (warp-opt)");
}
