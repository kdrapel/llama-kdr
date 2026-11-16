#pragma once

// Decode attention for q4_0 K / q4_0 V, head size 256, GQA 6, 1-8 query tokens, on RDNA4 WMMA.
// Split-K flash decoding over 64-token chunks; contiguous K/V only. Plain HIP (no ggml dependencies).
//
// Adapted from the q8_0 K / q4_0 V kernel in achyudh/llama.cpp (fattn-paged-decode-kernel.cuh): the K path reads
// q4_0 blocks, the page table, ring and cross-queue sync of that version are dropped.
//
// One block = one KV head x a range of chunks, 8 waves of 32. Rows are (query token, GQA head) pairs, packed into
// one or two 16-row WMMA tiles.
//
// Per chunk:
//   1. raw K and V rows go to LDS (16 B loads; the next chunk is prefetched into registers), the V scales also to a
//      token-contiguous array;
//   2. S = Q K^T with WMMA, K dequantized to f16 while building the B fragments;
//   3. online softmax by row through LDS (P as f16, per-row rescale factor);
//   4. O += P V with WMMA, V dequantized while gathering B fragments.
// A second kernel combines the splits.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define FA_QR_RDNA4 1
#else
#define FA_QR_RDNA4 0
#endif

static constexpr int FA_QR_D      = 256;
static constexpr int FA_QR_GQA    = 6;
static constexpr int FA_QR_CHUNK  = 64;
static constexpr int FA_QR_NWARPS = 8;
static constexpr int FA_QR_QK     = 32;
static constexpr int FA_QR_BLK    = 18;                                  // block_q4_0 bytes
static constexpr int FA_QR_ROW    = FA_QR_D / FA_QR_QK * FA_QR_BLK;      // 144 B
static constexpr int FA_QR_VEC    = FA_QR_ROW / 16;                      // 9 x 16 B
static constexpr int FA_QR_LOADS  = (FA_QR_CHUNK * FA_QR_VEC + FA_QR_NWARPS * 32 - 1) / (FA_QR_NWARPS * 32);

static_assert(FA_QR_ROW % 16 == 0, "rows must allow 16 B loads");

#if FA_QR_RDNA4
typedef _Float16 fa_qr_half8  __attribute__((ext_vector_type(8)));
typedef float    fa_qr_float8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ fa_qr_float8 fa_qr_wmma(const fa_qr_half8 a, const fa_qr_half8 b, const fa_qr_float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
#endif

// Two q4_0 nibbles as half2, each minus 8: 0x6400 is half(1024), and 1024 + n - 1032 = n - 8 is exact in f16.
static __device__ __forceinline__ half2 fa_qr_nib2(const uint32_t n0, const uint32_t n1) {
    const uint32_t u = 0x64006400u | n0 | (n1 << 16);
    return __hsub2(*(const half2 *) &u, __halves2half2(__float2half(1032.0f), __float2half(1032.0f)));
}

// M_TILES 16-row WMMA tiles over the rows of a KV head; MT bounds the query tokens one tile spans.
template <int M_TILES, int MT>
__launch_bounds__(FA_QR_NWARPS * 32, 6) // keeps every variant at <= 240 VGPRs: 3 blocks of 8 waves per WGP
static __global__ void fa_qr_partial(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const char * __restrict__ K, const int64_t k_nb1, const int64_t k_nb2,
        const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const char * __restrict__ mask, const int64_t mask_nb1,
        const float scale, const int n_kv, const int nb, const int n_chunks, const int chunks_per_split,
        float * __restrict__ O_part, float2 * __restrict__ ML_part) {
#if FA_QR_RDNA4
    constexpr int R   = M_TILES * 16;         // rows
    constexpr int TPR = FA_QR_NWARPS * 32 / R; // softmax threads per row
    constexpr int TOK = FA_QR_CHUNK / TPR;     // tokens per softmax thread

    __shared__ __align__(16) char     Kr[FA_QR_CHUNK * FA_QR_ROW + 16]; // pad: an aligned 8 B read may end past a row
    __shared__ __align__(16) char     Vr[FA_QR_CHUNK * FA_QR_ROW];
    __shared__ __align__(16) half     Vs[FA_QR_D / FA_QR_QK * FA_QR_CHUNK]; // V scales by block, then token
    __shared__ __align__(16) float    Ss[R * FA_QR_CHUNK];
    __shared__ __align__(16) _Float16 Ps[R * FA_QR_CHUNK];
    __shared__                float    alpha_s[R];
    __shared__ __align__(16) half     Ms[MT * FA_QR_CHUNK]; // mask rows of the chunk, one per query token of the tile

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int head = blockIdx.y;                     // KV head
    const int split = blockIdx.x;
    const int c0 = split * chunks_per_split;
    const int c1 = min(n_chunks, c0 + chunks_per_split);
    const int row0 = blockIdx.z * R;                 // first row of the tile
    const int tok0 = row0 / FA_QR_GQA;               // its query token
    const int n_rows = min(R, nb * FA_QR_GQA - row0); // valid rows of the tile

    // S phase: wave -> (m-tile, 16-token n-tile). Q fragments stay in registers.
    const int s_mt = warp / 4;
    const int s_nt = warp % 4;
    const bool s_active = s_mt < M_TILES;
    fa_qr_half8 Qa[FA_QR_D / 16];
    {
        const int r = s_mt * 16 + lane % 16;
        const bool valid = s_active && r < n_rows;
        const int g = row0 + r;
        const float4 * q = valid ? (const float4 *) (Q + (g / FA_QR_GQA) * q_nb1 + (head * FA_QR_GQA + g % FA_QR_GQA) * q_nb2) : nullptr;
#pragma unroll
        for (int ks = 0; ks < FA_QR_D / 16; ++ks) {
            const int d0 = ks * 16 + 8 * (lane / 16);
            const float4 lo = valid ? q[d0 / 4 + 0] : make_float4(0, 0, 0, 0);
            const float4 hi = valid ? q[d0 / 4 + 1] : make_float4(0, 0, 0, 0);
            Qa[ks] = fa_qr_half8{(_Float16) (lo.x * scale), (_Float16) (lo.y * scale), (_Float16) (lo.z * scale), (_Float16) (lo.w * scale),
                                 (_Float16) (hi.x * scale), (_Float16) (hi.y * scale), (_Float16) (hi.z * scale), (_Float16) (hi.w * scale)};
        }
    }

    // Softmax phase: TPR consecutive threads per row.
    const int sm_row = tid / TPR;
    const int sm_sub = tid % TPR;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // PV phase: wave -> 32 output dims.
    fa_qr_float8 O[M_TILES][2];
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            O[mt][dt] = fa_qr_float8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    uint4 kreg[FA_QR_LOADS];
    uint4 vreg[FA_QR_LOADS];
    // Mask values travel with K/V: a separate global read in the S phase would wait for the whole prefetch.
    constexpr int M_LOADS = MT * FA_QR_CHUNK / (FA_QR_NWARPS * 32);
    half mreg[M_LOADS];
    auto fetch = [&](const int c) {
#pragma unroll
        for (int j = 0; j < FA_QR_LOADS; ++j) {
            const int i = tid + j * FA_QR_NWARPS * 32;
            const int r = i / FA_QR_VEC, col = i % FA_QR_VEC;
            const int t = c * FA_QR_CHUNK + r;
            kreg[j] = i < FA_QR_CHUNK * FA_QR_VEC && t < n_kv ? ((const uint4 *) (K + (int64_t) t * k_nb1 + head * k_nb2))[col] : make_uint4(0, 0, 0, 0);
            vreg[j] = i < FA_QR_CHUNK * FA_QR_VEC && t < n_kv ? ((const uint4 *) (V + (int64_t) t * v_nb1 + head * v_nb2))[col] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < M_LOADS; ++j) {
            const int i = tid + j * FA_QR_NWARPS * 32;
            const int q = tok0 + i / FA_QR_CHUNK, t = c * FA_QR_CHUNK + i % FA_QR_CHUNK;
            mreg[j] = mask && q < nb && t < n_kv ? ((const half *) (mask + q * mask_nb1))[t] : __float2half(0.0f);
        }
    };

    if (c0 < c1) {
        fetch(c0);
    }
    for (int c = c0; c < c1; ++c) {
        // 1. registers -> LDS, then prefetch the next chunk.
#pragma unroll
        for (int j = 0; j < FA_QR_LOADS; ++j) {
            const int i = tid + j * FA_QR_NWARPS * 32;
            if (i < FA_QR_CHUNK * FA_QR_VEC) {
                ((uint4 *) Kr)[i] = kreg[j];
                ((uint4 *) Vr)[i] = vreg[j];
                // 16 B piece v of a row holds the scale of block v at byte 2 * v (v < 8; piece 8 holds none)
                const int r = i / FA_QR_VEC, v = i % FA_QR_VEC;
                if (v < FA_QR_D / FA_QR_QK) {
                    const uint32_t w = v < 2 ? vreg[j].x : v < 4 ? vreg[j].y : v < 6 ? vreg[j].z : vreg[j].w;
                    const uint16_t s = (uint16_t) (v % 2 ? w >> 16 : w);
                    Vs[v * FA_QR_CHUNK + r] = *(const half *) &s;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < M_LOADS; ++j) {
            Ms[tid + j * FA_QR_NWARPS * 32] = mreg[j];
        }
        __syncthreads();
        if (c + 1 < c1) {
            fetch(c + 1);
        }

        // 2. S = Q K^T for this wave's tile.
        if (s_active) {
            fa_qr_float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int n = s_nt * 16 + lane % 16;
            const char * krow = Kr + n * FA_QR_ROW;
#pragma unroll
            for (int ks = 0; ks < FA_QR_D / 16; ++ks) {
                // The fragment's 8 dims start at nibble o of block ks / 2 and span 8 bytes. A block's data starts at
                // byte 2, so that offset is not 8 B aligned: read from the aligned base and shift by a constant per
                // block, as the q8_0 version of this kernel does.
                const int blk = ks / 2;
                const int o   = (ks % 2) * 16 + 8 * (lane / 16);
                const int off = blk * FA_QR_BLK + 2 + (o & 15);
                const int sh  = off & 7;
                const uint64_t * w = (const uint64_t *) (krow + off - sh);
                const uint64_t raw = sh == 0 ? w[0] : (w[0] >> (8 * sh)) | (w[1] << (64 - 8 * sh));
                const uint32_t lo = (uint32_t) raw, hi = (uint32_t) (raw >> 32);
                const int   shn = o >= 16 ? 4 : 0; // q4_0: low nibbles are dims 0-15, high nibbles 16-31
                const half  d   = *(const half *) (krow + blk * FA_QR_BLK);
                const half2 d2  = __halves2half2(d, d);
                half2 b2[4];
                b2[0] = __hmul2(fa_qr_nib2((lo >> (0  + shn)) & 0x0F, (lo >> (8  + shn)) & 0x0F), d2);
                b2[1] = __hmul2(fa_qr_nib2((lo >> (16 + shn)) & 0x0F, (lo >> (24 + shn)) & 0x0F), d2);
                b2[2] = __hmul2(fa_qr_nib2((hi >> (0  + shn)) & 0x0F, (hi >> (8  + shn)) & 0x0F), d2);
                b2[3] = __hmul2(fa_qr_nib2((hi >> (16 + shn)) & 0x0F, (hi >> (24 + shn)) & 0x0F), d2);
                acc = fa_qr_wmma(Qa[ks], *(const fa_qr_half8 *) b2, acc);
            }
            const int t_kv = c * FA_QR_CHUNK + n;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r = s_mt * 16 + 8 * (lane / 16) + l;
                Ss[r * FA_QR_CHUNK + n] = r < n_rows && t_kv < n_kv ? acc[l] + __half2float(Ms[((row0 + r) / FA_QR_GQA - tok0) * FA_QR_CHUNK + n]) : -INFINITY;
            }
        }
        __syncthreads();

        // 3. Online softmax by row.
        {
            const float * srow = Ss + sm_row * FA_QR_CHUNK + sm_sub * TOK;
            float s[TOK];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < TOK; ++j) {
                s[j] = srow[j];
                mx = fmaxf(mx, s[j]);
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                mx = fmaxf(mx, __shfl_xor(mx, o, 32));
            }
            const float m_new = fmaxf(m_run, mx);
            float sum = 0.0f;
            _Float16 * prow = Ps + sm_row * FA_QR_CHUNK + sm_sub * TOK;
#pragma unroll
            for (int j = 0; j < TOK; ++j) {
                const float p = m_new == -INFINITY ? 0.0f : __expf(s[j] - m_new);
                sum += p;
                prow[j] = (_Float16) p;
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                sum += __shfl_xor(sum, o, 32);
            }
            const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_run - m_new);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if (sm_sub == 0) {
                alpha_s[sm_row] = alpha;
            }
        }
        __syncthreads();

        // 4. O = O * alpha + P V for this wave's 32 dims.
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float a = alpha_s[mt * 16 + 8 * (lane / 16) + l];
                O[mt][0][l] *= a;
                O[mt][1][l] *= a;
            }
        }
#pragma unroll
        for (int ks = 0; ks < FA_QR_CHUNK / 16; ++ks) {
            const int k0 = ks * 16 + 8 * (lane / 16);
            fa_qr_half8 a[M_TILES];
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                a[mt] = *(const fa_qr_half8 *) (Ps + (mt * 16 + lane % 16) * FA_QR_CHUNK + k0);
            }
            // The wave's 32 dims are block `warp`: byte lane % 16 holds dims lane % 16 (low nibble, dt 0) and
            // lane % 16 + 16 (high nibble, dt 1). The scales of the 8 tokens are one read of Vs.
            uint32_t xb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const char * blk0 = Vr + (k0 + 2*j + 0) * FA_QR_ROW + warp * FA_QR_BLK;
                const char * blk1 = Vr + (k0 + 2*j + 1) * FA_QR_ROW + warp * FA_QR_BLK;
                xb[j] = (uint32_t) (uint8_t) blk0[2 + lane % 16] | ((uint32_t) (uint8_t) blk1[2 + lane % 16] << 8);
            }
            const uint4 sv = *(const uint4 *) (Vs + warp * FA_QR_CHUNK + k0);
            const half2 s2[4] = { *(const half2 *) &sv.x, *(const half2 *) &sv.y, *(const half2 *) &sv.z, *(const half2 *) &sv.w };
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) {
                const int shift = dt * 4;
                half2 b2[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t q0 = (xb[j] >> shift) & 0x0F;
                    const uint32_t q1 = (xb[j] >> (8 + shift)) & 0x0F;
                    b2[j] = __hmul2(fa_qr_nib2(q0, q1), s2[j]);
                }
#pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    O[mt][dt] = fa_qr_wmma(a[mt], *(const fa_qr_half8 *) b2, O[mt][dt]);
                }
            }
        }
        __syncthreads();
    }

    // Partial results for valid rows: unnormalized O plus (max, sum) per row.
    const int64_t base = (((int64_t) split * gridDim.y + head) * gridDim.z + blockIdx.z) * R;
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            const int dim = warp * 32 + dt * 16 + lane % 16;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r = mt * 16 + 8 * (lane / 16) + l;
                if (r < n_rows) {
                    O_part[(base + r) * FA_QR_D + dim] = O[mt][dt][l];
                }
            }
        }
    }
    if (sm_sub == 0 && sm_row < n_rows) {
        ML_part[base + sm_row] = make_float2(m_run, l_run);
    }
#endif // FA_QR_RDNA4
}

// One block per (row, KV head), one thread per output dim.
static __global__ void fa_qr_combine(
        const float * __restrict__ O_part, const float2 * __restrict__ ML_part, const int n_splits, const int R,
        char * __restrict__ dst, const int64_t dst_nb1, const int64_t dst_nb2) {
    const int r = blockIdx.x;                 // (query token, GQA head)
    const int head = blockIdx.y;
    const int n_heads = gridDim.y;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        M = fmaxf(M, ML_part[((int64_t) s * n_heads + head) * R + r].x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const int64_t i = ((int64_t) s * n_heads + head) * R + r;
        const float2 ml = ML_part[i];
        const float w = ml.x == -INFINITY ? 0.0f : expf(ml.x - M);
        num += w * O_part[i * FA_QR_D + threadIdx.x];
        den += w * ml.y;
    }
    float * out = (float *) (dst + (r / FA_QR_GQA) * dst_nb2 + (head * FA_QR_GQA + r % FA_QR_GQA) * dst_nb1);
    out[threadIdx.x] = den == 0.0f ? 0.0f : num / den; // a fully masked row is 0, as in the CPU reference
}
