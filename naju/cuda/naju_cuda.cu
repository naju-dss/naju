// Naju v1 scan — native CUDA kernels (forward + recompute backward).
//
// Recurrence (per batch b, inner channel d, state index n), matching the
// sequential reference implementation (naju/scan.py):
//
//   f_t = sigmoid(f_logit[b,t,d])          (per (b,t,d) scalar, broadcast over n)
//   i_t = sigmoid(i_logit[b,t,d])
//   x[b,d,n] = f_t * x[b,d,n] + i_t * B[b,t,n] * u[b,t,d]
//   y[b,t,d] = sum_n C[b,t,n] * x[b,d,n] + D[d] * u[b,t,d]
//
// Mapping: grid = (B, D_inner), block = next_pow2(N) threads, thread n owns
// state x[n] in a register across the whole time loop (never materialized to
// HBM). y is reduced across threads. For training the forward checkpoints the
// state every CK steps into `ckpt` ([B, D, nchunks, N]); the backward recomputes
// each chunk forward from its checkpoint (Mamba-style recompute) so the full
// [B,T,D,N] state history is never stored.
//
// All tensors are float32, contiguous. d_state N <= 128.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

constexpr int CK = 32;  // recompute chunk length

__device__ __forceinline__ float sigf(float x) { return 1.f / (1.f + __expf(-x)); }

static inline int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

// ─────────────────────────────── forward ────────────────────────────────────
__global__ void naju_fwd_kernel(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    float* __restrict__ y, float* __restrict__ ckpt,
    int B, int T, int D, int N, int nchunks, bool save_ckpt) {
  const int b = blockIdx.x, d = blockIdx.y, n = threadIdx.x, BT = blockDim.x;
  const bool act = n < N;
  extern __shared__ float sh[];                     // size BT
  const float Dval = Dd[d];
  float s = 0.f;

  for (int t = 0; t < T; ++t) {
    if (save_ckpt && (t % CK == 0) && act)
      ckpt[((((long)b * D + d) * nchunks) + t / CK) * N + n] = s;   // x_{t-1}
    const long btd = ((long)b * T + t) * D + d;
    const long btn = ((long)b * T + t) * N + n;
    const float u_t = u[btd];
    const float f = sigf(fl[btd]);
    const float i = sigf(il[btd]);
    const float Bt = act ? Bm[btn] : 0.f;
    const float Ct = act ? Cm[btn] : 0.f;
    s = f * s + i * Bt * u_t;                        // x_t
    sh[n] = act ? Ct * s : 0.f;
    __syncthreads();
    for (int st = BT >> 1; st > 0; st >>= 1) {
      if (n < st) sh[n] += sh[n + st];
      __syncthreads();
    }
    if (n == 0) y[btd] = sh[0] + Dval * u_t;
    __syncthreads();
  }
}

// ─────────────────────────────── backward ───────────────────────────────────
// Shared layout: [ xcur : CK*N floats ][ red : BT floats ]
__global__ void naju_bwd_kernel(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    const float* __restrict__ ckpt, const float* __restrict__ dy,
    float* __restrict__ du, float* __restrict__ df, float* __restrict__ di,
    float* __restrict__ dB, float* __restrict__ dC, float* __restrict__ dDacc,
    int B, int T, int D, int N, int nchunks) {
  const int b = blockIdx.x, d = blockIdx.y, n = threadIdx.x, BT = blockDim.x;
  const bool act = n < N;
  extern __shared__ float sh[];
  float* xcur = sh;             // CK*N
  float* red  = sh + CK * N;    // BT
  const float Dval = Dd[d];
  float carry = 0.f;            // d x_t flowing from the future (per n)
  float dDsum = 0.f;

  for (int c = nchunks - 1; c >= 0; --c) {
    const int t0 = c * CK;
    const int len = min(CK, T - t0);
    // recompute forward within chunk from the checkpoint (x_{t0-1})
    float s = act ? ckpt[((((long)b * D + d) * nchunks) + c) * N + n] : 0.f;
    const float start_state = s;
    for (int tl = 0; tl < len; ++tl) {
      const int t = t0 + tl;
      const long btd = ((long)b * T + t) * D + d;
      const long btn = ((long)b * T + t) * N + n;
      const float u_t = u[btd];
      const float f = sigf(fl[btd]);
      const float i = sigf(il[btd]);
      const float Bt = act ? Bm[btn] : 0.f;
      s = f * s + i * Bt * u_t;              // x_t
      if (act) xcur[tl * N + n] = s;
    }
    __syncthreads();
    // reverse over the chunk
    for (int tl = len - 1; tl >= 0; --tl) {
      const int t = t0 + tl;
      const long btd = ((long)b * T + t) * D + d;
      const long btn = ((long)b * T + t) * N + n;
      const float u_t = u[btd];
      const float f = sigf(fl[btd]);
      const float i = sigf(il[btd]);
      const float Bt = act ? Bm[btn] : 0.f;
      const float Ct = act ? Cm[btn] : 0.f;
      const float x_t    = act ? xcur[tl * N + n] : 0.f;
      const float x_prev = act ? (tl > 0 ? xcur[(tl - 1) * N + n] : start_state) : 0.f;
      const float dy_t = dy[btd];
      const float d_x = Ct * dy_t + carry;   // per n

      if (act) atomicAdd(&dC[btn], dy_t * x_t);
      dDsum += dy_t * u_t;                    // n-independent; thread0 stores
      carry = f * d_x;                        // -> x_{t-1}

      // reduce d_f = sum_n d_x*x_prev
      red[n] = act ? d_x * x_prev : 0.f;
      __syncthreads();
      for (int st = BT >> 1; st > 0; st >>= 1) { if (n < st) red[n] += red[n + st]; __syncthreads(); }
      const float d_f = red[0];
      __syncthreads();
      // reduce sumdB = sum_n d_x*B
      red[n] = act ? d_x * Bt : 0.f;
      __syncthreads();
      for (int st = BT >> 1; st > 0; st >>= 1) { if (n < st) red[n] += red[n + st]; __syncthreads(); }
      const float sumdB = red[0];
      __syncthreads();

      if (n == 0) {
        df[btd] = d_f * f * (1.f - f);
        di[btd] = (sumdB * u_t) * i * (1.f - i);
        du[btd] = dy_t * Dval + i * sumdB;
      }
      if (act) atomicAdd(&dB[btn], d_x * i * u_t);
      __syncthreads();
    }
  }
  if (n == 0) dDacc[(long)b * D + d] = dDsum;
}

// ─────────────── chunk-parallel forward (inference) ─────────────────────────
// Parallelizes the time axis: grid (B, D, G) chunks run concurrently, giving
// B*D*G blocks instead of B*D, so latency stays hidden at small batch.
constexpr int CKP = 128;  // chunk length for the parallel scan

// Phase 1: per-chunk local scan from zero init -> chunk f-product Pf, end-state S.
__global__ void chunk_p1(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    float* __restrict__ Pf, float* __restrict__ S,
    int B, int T, int D, int N, int G) {
  const int b = blockIdx.x, d = blockIdx.y, c = blockIdx.z, n = threadIdx.x;
  const bool act = n < N;
  const int t0 = c * CKP, len = min(CKP, T - t0);
  float s = 0.f, pf = 1.f;
  for (int tl = 0; tl < len; ++tl) {
    const int t = t0 + tl;
    const long btd = ((long)b * T + t) * D + d;
    const float f = sigf(fl[btd]);
    const float v = sigf(il[btd]) * (act ? Bm[((long)b * T + t) * N + n] : 0.f) * u[btd];
    s = f * s + v;
    pf *= f;
  }
  if (act) S[(((long)b * D + d) * G + c) * N + n] = s;
  if (n == 0) Pf[((long)b * D + d) * G + c] = pf;
}

// Phase 2: sequential carry across chunks (G steps), independent per n (no sync).
__global__ void chunk_p2(
    const float* __restrict__ Pf, const float* __restrict__ S,
    float* __restrict__ Init, int B, int D, int N, int G) {
  const int b = blockIdx.x, d = blockIdx.y, n = threadIdx.x;
  if (n >= N) return;
  float init = 0.f;
  for (int c = 0; c < G; ++c) {
    Init[(((long)b * D + d) * G + c) * N + n] = init;      // state entering chunk c
    init = Pf[((long)b * D + d) * G + c] * init + S[(((long)b * D + d) * G + c) * N + n];
  }
}

// Phase 3: per-chunk local scan from the true init state -> y.
__global__ void chunk_p3(
    const float* __restrict__ u,  const float* __restrict__ fl,
    const float* __restrict__ il, const float* __restrict__ Bm,
    const float* __restrict__ Cm, const float* __restrict__ Dd,
    const float* __restrict__ Init, float* __restrict__ y,
    int B, int T, int D, int N, int G) {
  const int b = blockIdx.x, d = blockIdx.y, c = blockIdx.z, n = threadIdx.x, BT = blockDim.x;
  const bool act = n < N;
  extern __shared__ float sh[];
  const int t0 = c * CKP, len = min(CKP, T - t0);
  const float Dval = Dd[d];
  float s = act ? Init[(((long)b * D + d) * G + c) * N + n] : 0.f;
  for (int tl = 0; tl < len; ++tl) {
    const int t = t0 + tl;
    const long btd = ((long)b * T + t) * D + d;
    const long btn = ((long)b * T + t) * N + n;
    const float u_t = u[btd];
    const float f = sigf(fl[btd]);
    const float Bt = act ? Bm[btn] : 0.f;
    const float Ct = act ? Cm[btn] : 0.f;
    s = f * s + sigf(il[btd]) * Bt * u_t;
    sh[n] = act ? Ct * s : 0.f;
    __syncthreads();
    for (int st = BT >> 1; st > 0; st >>= 1) { if (n < st) sh[n] += sh[n + st]; __syncthreads(); }
    if (n == 0) y[btd] = sh[0] + Dval * u_t;
    __syncthreads();
  }
}

torch::Tensor naju_fwd_chunked(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kFloat32, "float32 CUDA tensors required");
  const int Bsz = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  TORCH_CHECK(N <= 128, "d_state must be <= 128");
  const int G = (T + CKP - 1) / CKP;
  auto opt = u.options();
  auto y = torch::empty({Bsz, T, D}, opt);
  auto Pf = torch::empty({Bsz, D, G}, opt);
  auto S = torch::empty({Bsz, D, G, N}, opt);
  auto Init = torch::empty({Bsz, D, G, N}, opt);
  const int BT = std::max(next_pow2(N), 32);
  chunk_p1<<<dim3(Bsz, D, G), BT>>>(u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Pf.data_ptr<float>(), S.data_ptr<float>(), Bsz, T, D, N, G);
  chunk_p2<<<dim3(Bsz, D), BT>>>(Pf.data_ptr<float>(), S.data_ptr<float>(),
      Init.data_ptr<float>(), Bsz, D, N, G);
  chunk_p3<<<dim3(Bsz, D, G), BT, BT * sizeof(float)>>>(u.data_ptr<float>(), fl.data_ptr<float>(),
      il.data_ptr<float>(), Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      Init.data_ptr<float>(), y.data_ptr<float>(), Bsz, T, D, N, G);
  return y;
}

// ─────────────────────────────── launchers ──────────────────────────────────
std::vector<torch::Tensor> naju_fwd(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd, bool save_ckpt) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kFloat32, "float32 CUDA tensors required");
  const int Bsz = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  TORCH_CHECK(N <= 128, "d_state must be <= 128");
  const int nchunks = (T + CK - 1) / CK;
  auto opt = u.options();
  auto y = torch::empty({Bsz, T, D}, opt);
  auto ckpt = save_ckpt ? torch::empty({Bsz, D, nchunks, N}, opt) : torch::empty({0}, opt);
  const int BT = std::max(next_pow2(N), 32);
  dim3 grid(Bsz, D);
  naju_fwd_kernel<<<grid, BT, BT * sizeof(float)>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      y.data_ptr<float>(), save_ckpt ? ckpt.data_ptr<float>() : nullptr,
      Bsz, T, D, N, nchunks, save_ckpt);
  return {y, ckpt};
}

std::vector<torch::Tensor> naju_bwd(
    torch::Tensor u, torch::Tensor fl, torch::Tensor il,
    torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dd,
    torch::Tensor ckpt, torch::Tensor dy) {
  const int Bsz = u.size(0), T = u.size(1), D = u.size(2), N = Bm.size(2);
  const int nchunks = (T + CK - 1) / CK;
  auto opt = u.options();
  auto du = torch::empty_like(u);
  auto df = torch::empty_like(fl);
  auto di = torch::empty_like(il);
  auto dB = torch::zeros_like(Bm);
  auto dC = torch::zeros_like(Cm);
  auto dDacc = torch::zeros({Bsz, D}, opt);
  const int BT = std::max(next_pow2(N), 32);
  const size_t shmem = ((size_t)CK * N + BT) * sizeof(float);
  dim3 grid(Bsz, D);
  naju_bwd_kernel<<<grid, BT, shmem>>>(
      u.data_ptr<float>(), fl.data_ptr<float>(), il.data_ptr<float>(),
      Bm.data_ptr<float>(), Cm.data_ptr<float>(), Dd.data_ptr<float>(),
      ckpt.data_ptr<float>(), dy.data_ptr<float>(),
      du.data_ptr<float>(), df.data_ptr<float>(), di.data_ptr<float>(),
      dB.data_ptr<float>(), dC.data_ptr<float>(), dDacc.data_ptr<float>(),
      Bsz, T, D, N, nchunks);
  auto dD = dDacc.sum(0);
  return {du, df, di, dB, dC, dD};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &naju_fwd, "Naju v1 forward (CUDA, sequential + ckpt)");
  m.def("bwd", &naju_bwd, "Naju v1 recompute backward (CUDA)");
  m.def("fwd_chunked", &naju_fwd_chunked, "Naju v1 chunk-parallel forward (inference)");
}
