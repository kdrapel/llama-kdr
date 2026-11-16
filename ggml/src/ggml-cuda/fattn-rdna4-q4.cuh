#pragma once

#include "common.cuh"

// Decode attention for q4_0 K / q4_0 V with head size 256, GQA 6 and 1-8 query tokens (speculative verify) on RDNA4.
// Split-K flash decoding with WMMA and in-register dequantization, contiguous K/V only. GGML_CUDA_DISABLE_FA_QR
// falls back to the stock kernels.
bool ggml_cuda_fattn_rdna4_q4_enabled(int device, const ggml_tensor * dst);
size_t ggml_cuda_fattn_rdna4_q4_alloc_size(int device, const ggml_tensor * dst); // dst and its split partials
void ggml_cuda_flash_attn_ext_rdna4_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
