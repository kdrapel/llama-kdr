#include "fattn-rdna4-q4.cuh"

#if defined(GGML_USE_HIP)

#include "fattn-rdna4-q4-kernel.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

static bool fa_qr_disabled() {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_FA_QR") != nullptr;
    return disabled;
}

// Chunks of K/V, split over the blocks of one resident wave.
struct fa_qr_shape {
    int n_kv, nb, n_heads, M_TILES, R, n_tiles, n_chunks, n_splits, chunks_per_split;
    size_t dst_size, o_size, ml_size;

    fa_qr_shape(int device, const ggml_tensor * dst) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];

        n_kv     = (int) K->ne[1];
        nb       = (int) Q->ne[1];
        n_heads  = (int) K->ne[2];
        M_TILES  = nb * FA_QR_GQA <= 16 ? 1 : 2;
        R        = M_TILES * 16;
        n_tiles  = (nb * FA_QR_GQA + R - 1) / R;
        n_chunks = (n_kv + FA_QR_CHUNK - 1) / FA_QR_CHUNK;

        // One resident wave of blocks: three 8-wave blocks per WGP (nsm counts WGPs on RDNA4).
        const int want = 3 * ggml_cuda_info().devices[device].nsm;
        n_splits = std::max(1, std::min(n_chunks, (want + n_heads * n_tiles - 1) / (n_heads * n_tiles)));
        chunks_per_split = (n_chunks + n_splits - 1) / n_splits;
        n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

        dst_size = GGML_PAD(ggml_nbytes(dst), 256);
        o_size   = GGML_PAD((size_t) n_splits * n_heads * n_tiles * R * FA_QR_D * sizeof(float), 256);
        ml_size  = GGML_PAD((size_t) n_splits * n_heads * n_tiles * R * sizeof(float2), 256);
    }

    size_t total() const { return dst_size + o_size + ml_size; }
};

bool ggml_cuda_fattn_rdna4_q4_enabled(int device, const ggml_tensor * dst) {
    if (fa_qr_disabled() || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[device].cc)) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!Q || !K || !V) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (Q->ne[0] != FA_QR_D || K->ne[0] != FA_QR_D || V->ne[0] != FA_QR_D) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > 8 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[1] < 1 || K->ne[2] < 1 || Q->ne[2] != FA_QR_GQA * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) {
        return false;
    }
    // sinks, row lists and block lists take the stock paths
    for (int i = 4; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i]) {
            return false;
        }
    }

    float params[3];
    memcpy(params, dst->op_params, sizeof(params)); // scale, max_bias, logit_softcap
    if (params[1] != 0.0f || params[2] != 0.0f) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return false;
    }
    // 16 B loads and the 8 B aligned fragment reads
    if (K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    return true;
}

size_t ggml_cuda_fattn_rdna4_q4_alloc_size(int device, const ggml_tensor * dst) {
    return fa_qr_shape(device, dst).total();
}

void ggml_cuda_flash_attn_ext_rdna4_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const fa_qr_shape sh(ctx.device, dst);
    if (getenv("GGML_CUDA_FA_QR_LOG")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[fa_qr] n_kv=%d nb=%d tiles=%d splits=%d\n", sh.n_kv, sh.nb, sh.n_tiles, sh.n_splits);
        }
    }
    const int n_tiles = sh.n_tiles;

    // reserved after dst by ggml_cuda_fattn_rdna4_q4_alloc_size()
    float  * O_part  = (float *)  ((char *) dst->data + sh.dst_size);
    float2 * ML_part = (float2 *) ((char *) O_part + sh.o_size);

    const dim3 grid(sh.n_splits, sh.n_heads, n_tiles);
    const char * mask_data = mask ? (const char *) mask->data : nullptr;
    const int64_t mask_nb1 = mask ? mask->nb[1] : 0;

#define FA_QR_LAUNCH(MTILES, MTOK) \
    fa_qr_partial<MTILES, MTOK><<<grid, FA_QR_NWARPS * 32, 0, stream>>>( \
        (const char *) Q->data, Q->nb[1], Q->nb[2], \
        (const char *) K->data, K->nb[1], K->nb[2], \
        (const char *) V->data, V->nb[1], V->nb[2], \
        mask_data, mask_nb1, scale, sh.n_kv, sh.nb, sh.n_chunks, sh.chunks_per_split, O_part, ML_part)

    if (n_tiles > 1) {
        FA_QR_LAUNCH(2, 8);  // a 32-row tile spans at most 7 query tokens
    } else if (sh.M_TILES == 1) {
        FA_QR_LAUNCH(1, 4);
    } else if (sh.nb <= 4) {
        FA_QR_LAUNCH(2, 4);
    } else {
        FA_QR_LAUNCH(2, 8);  // 5 query tokens (30 rows) in one 32-row tile with the 8-token mask staging
    }
#undef FA_QR_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    fa_qr_combine<<<dim3(sh.nb * FA_QR_GQA, sh.n_heads), FA_QR_D, 0, stream>>>(
        O_part, ML_part, sh.n_splits, n_tiles * sh.R, (char *) dst->data, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_fattn_rdna4_q4_enabled(int device, const ggml_tensor * dst) {
    GGML_UNUSED_VARS(device, dst);
    return false;
}

size_t ggml_cuda_fattn_rdna4_q4_alloc_size(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    return ggml_nbytes(dst);
}

void ggml_cuda_flash_attn_ext_rdna4_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("the RDNA4 q4_0 decode kernel is only built for HIP");
}

#endif // defined(GGML_USE_HIP)
