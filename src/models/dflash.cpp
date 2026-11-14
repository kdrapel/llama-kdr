#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;
    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_dspine   = ml.get_tensor_meta("dspine_codebook.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    if (has_markov) {
        variant = VARIANT_DSPARK;
    } else if (has_selector || has_dspine) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:
            if (!ml->get_tensor_meta("dspine_codebook.weight")) {
                load_tensors_dflash2();
            }
            break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DFLASH:                                       break;
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }

    const struct ggml_tensor * dspine_codebook_meta  = ml->get_tensor_meta("dspine_codebook.weight");
    const struct ggml_tensor * dspine_gate_meta      = ml->get_tensor_meta("dspine_gate.weight");
    const struct ggml_tensor * dspine_gate_b_meta    = ml->get_tensor_meta("dspine_gate.bias");
    const struct ggml_tensor * dspine_pred_gate_meta = ml->get_tensor_meta("dspine_pred_gate.weight");
    const struct ggml_tensor * dspine_prev_meta     = ml->get_tensor_meta("dspine_prev.weight");
    const bool has_dspine = dspine_codebook_meta != nullptr;
    if (has_dspine) {
        const int64_t code  = dspine_codebook_meta->ne[0];
        const int64_t rank  = dspine_gate_meta ? dspine_gate_meta->ne[1] : 0;
        const int64_t top_k = hparams.dflash_selector_top_k;
        if (!dspine_gate_meta || !dspine_gate_b_meta || !dspine_pred_gate_meta || !dspine_prev_meta ||
                ml->get_tensor_meta("selector_hidden.weight") || ml->get_tensor_meta("markov_w1.weight") ||
                n_vocab_draft != n_vocab || code <= 0 || dspine_codebook_meta->ne[1] != n_vocab || rank <= 0 ||
                dspine_gate_meta->ne[0] != code || dspine_gate_b_meta->ne[0] != rank ||
                dspine_pred_gate_meta->ne[0] != code || dspine_pred_gate_meta->ne[1] != rank ||
                dspine_prev_meta->ne[0] != code || dspine_prev_meta->ne[1] != rank ||
                top_k < 2 || top_k > n_embd || top_k > n_embd / (top_k + 1) || hparams.dflash_block_size < 2 ||
                backbone != BACKBONE_BASE) {
            throw std::runtime_error("invalid DSpine metadata");
        }

        dspine_codebook  = create_tensor(tn(LLM_TENSOR_DSPINE_CODEBOOK,  "weight"), { code, n_vocab }, 0);
        dspine_gate      = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "weight"), { code, rank }, 0);
        dspine_gate_b    = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "bias"),   { rank }, 0);
        dspine_pred_gate = create_tensor(tn(LLM_TENSOR_DSPINE_PRED_GATE, "weight"), { code, rank }, 0);
        dspine_prev      = create_tensor(tn(LLM_TENSOR_DSPINE_PREV,      "weight"), { code, rank }, 0);
        for (int i = 0; i < n_layer; ++i) {
            layers[i].dspine_read  = create_tensor(tn(LLM_TENSOR_DSPINE_READ,  "weight", i), { n_embd, code }, 0);
            layers[i].dspine_write = create_tensor(tn(LLM_TENSOR_DSPINE_WRITE, "weight", i), { rank, n_embd }, 0);
        }

        LLAMA_LOG_INFO("%s: DSpine code bridge (code = %lld, rank = %lld, top-k = %lld)\n", __func__,
                (long long) code, (long long) rank, (long long) top_k);
    } else if (dspine_gate_meta || dspine_gate_b_meta || dspine_pred_gate_meta || dspine_prev_meta) {
        throw std::runtime_error("incomplete DSpine tensors");
    }
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    const int64_t n_embd_inp = g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = g.build_lora_mm(model.fc, inp_target, model.fc_s);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

struct dspine_stash {
    ggml_tensor * raw = nullptr;
    ggml_tensor * rms = nullptr;
    ggml_tensor * q   = nullptr;
};

static ggml_tensor * dspine_anchor_ids(llm_graph_context & g, ggml_tensor * tokens, int64_t block, int64_t n_blocks) {
    return ggml_cont_1d(g.ctx0, ggml_view_2d(g.ctx0, tokens, 1, n_blocks, block * tokens->nb[0], 0), n_blocks);
}

static ggml_tensor * build_dspine_write(llm_graph_context & g, const llama_model & model,
        ggml_tensor * cur, ggml_tensor * tokens, int il, dspine_stash * stash) {
    ggml_context * ctx0 = g.ctx0;
    const auto & layer = model.layers[il];
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t code     = model.dspine_codebook->ne[0];
    const int64_t rank     = model.dspine_gate->ne[1];
    const float eps = g.hparams.f_norm_rms_eps;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    const int64_t block = n_tokens / n_blocks;
    if (block < 2) {
        return cur;
    }

    ggml_tensor * rms = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)), 1.0f/n_embd));
    ggml_tensor * codes = g.build_lora_mm(layer.dspine_read, ggml_rms_norm(ctx0, cur, eps));
    codes = ggml_scale(ctx0, ggml_l2_norm(ctx0, codes, eps), sqrtf((float) code));
    g.cb(codes, "dspine_codes", il);

    ggml_tensor * anchor_codes = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(g, tokens, block, n_blocks));
    anchor_codes = ggml_scale(ctx0, ggml_reshape_3d(ctx0, anchor_codes, code, 1, n_blocks), sqrtf((float) code));

    codes = ggml_reshape_3d(ctx0, codes, code, block, n_blocks);
    codes = ggml_concat(ctx0, anchor_codes,
            ggml_cont(ctx0, ggml_view_3d(ctx0, codes, code, block - 1, n_blocks, codes->nb[1], codes->nb[2], codes->nb[1])), 1);
    codes = ggml_reshape_2d(ctx0, codes, code, n_tokens);

    ggml_tensor * q = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_gate,      codes), rank, block, n_blocks);
    ggml_tensor * s = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_pred_gate, codes), rank, block, n_blocks);
    ggml_tensor * v = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_prev,      codes), rank, block, n_blocks);
    const auto rows = [&](ggml_tensor * t, int64_t beg) {
        return ggml_view_3d(ctx0, t, rank, block - 1, n_blocks, t->nb[1], t->nb[2], beg * t->nb[1]);
    };

    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, rows(q, 1), rows(s, 0)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, rows(v, 0));
    ggml_tensor * delta = g.build_lora_mm(layer.dspine_write, ggml_reshape_2d(ctx0, msg, rank, (block - 1) * n_blocks));
    delta = ggml_pad_ext(ctx0, ggml_reshape_3d(ctx0, delta, n_embd, block - 1, n_blocks), 0, 0, 1, 0, 0, 0, 0, 0);
    delta = ggml_mul(ctx0, ggml_reshape_2d(ctx0, delta, n_embd, n_tokens), rms);

    if (stash) {
        stash->raw = cur;
        stash->rms = rms;
        stash->q   = ggml_reshape_2d(ctx0, q, rank, n_tokens);
    }

    return ggml_add(ctx0, cur, delta);
}

static void build_dspine_lattice(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens,
        ggml_tensor * output, ggml_tensor * raw, ggml_tensor * rms, ggml_tensor * q) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;
    const int64_t n_embd   = g.n_embd;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t top_k    = g.hparams.dflash_selector_top_k;
    const int64_t code     = model.dspine_codebook->ne[0];
    const int64_t rank     = model.dspine_gate->ne[1];

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    const int64_t block = n_tokens / n_blocks;
    const int64_t n_pos = block - 1;
    if (block < 2 || block > (int64_t) g.hparams.dflash_block_size || raw == nullptr) {
        return;
    }

    const auto pos_rows = [&](ggml_tensor * t) {
        ggml_tensor * t3 = ggml_reshape_3d(ctx0, t, t->ne[0], block, n_blocks);
        t3 = ggml_cont(ctx0, ggml_view_3d(ctx0, t3, t->ne[0], n_pos, n_blocks, t3->nb[1], t3->nb[2], t3->nb[1]));
        return ggml_reshape_4d(ctx0, t3, t->ne[0], 1, n_pos, n_blocks);
    };

    ggml_tensor * cand = ggml_reshape_3d(ctx0, ggml_top_k(ctx0, res->t_logits, top_k), top_k, block, n_blocks);
    ggml_tensor * cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand, top_k, n_pos, n_blocks,
            cand->nb[1], cand->nb[2], cand->nb[1]));

    ggml_tensor * pred = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(g, tokens, block, n_blocks));
    pred = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, pred, code, 1, 1, n_blocks), code, top_k, 1, n_blocks);
    if (n_pos > 1) {
        ggml_tensor * prev_ids = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run, top_k, n_pos - 1,
                n_blocks, cand_run->nb[1], cand_run->nb[2], 0));
        ggml_tensor * prev = ggml_get_rows(ctx0, model.dspine_codebook,
                ggml_reshape_1d(ctx0, prev_ids, top_k * (n_pos - 1) * n_blocks));
        pred = ggml_concat(ctx0, pred, ggml_reshape_4d(ctx0, prev, code, top_k, n_pos - 1, n_blocks), 2);
    }
    pred = ggml_scale(ctx0, ggml_reshape_2d(ctx0, pred, code, top_k * n_pos * n_blocks), sqrtf((float) code));

    ggml_tensor * gp = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_pred_gate, pred), rank, top_k, n_pos, n_blocks);
    ggml_tensor * pv = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_prev, pred), rank, top_k, n_pos, n_blocks);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, gp, pos_rows(q)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, pv);

    const auto & last = model.layers[g.n_layer - 1];
    ggml_tensor * delta = g.build_lora_mm(last.dspine_write,
            ggml_reshape_2d(ctx0, msg, rank, top_k * n_pos * n_blocks));
    delta = ggml_reshape_4d(ctx0, delta, n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * refined = ggml_add(ctx0, ggml_mul(ctx0, delta, pos_rows(rms)), pos_rows(raw));
    refined = g.build_norm(refined, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    g.cb(refined, "dspine_refined", -1);

    ggml_tensor * head = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand_run, top_k * n_pos * n_blocks));
    head = ggml_reshape_4d(ctx0, head, n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * table = ggml_mul_mat(ctx0, head, refined);
    ggml_tensor * row = ggml_concat(ctx0,
            ggml_cast(ctx0, cand_run, GGML_TYPE_F32),
            ggml_reshape_3d(ctx0, table, top_k * top_k, n_pos, n_blocks), 0);
    row = ggml_pad(ctx0, row, n_embd - top_k - top_k * top_k, 0, 0, 0);

    ggml_tensor * packed = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0f);
    packed = ggml_reshape_2d(ctx0, ggml_concat(ctx0, packed, row, 1), n_embd, n_tokens);
    g.cb(packed, "dspine_lattice", -1);
    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos) {
    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, inp_g, layer.wv_s);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        if (use_iswa) {
            // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            // rotate K/V into the cache's rotated space
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
            if (k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        } else {
            // rotate K/V into the cache's rotated space
            if (inp_attn->self_k_rot[il]) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
            }
            if (inp_attn->self_v_rot[il]) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inpL = ggml_get_rows(ctx0, get_tok_embd(*this, model), inp_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    dspine_stash dspine;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos);
        Kcur = build_rope(Kcur, inp_pos);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        if (layer.dspine_write) {
            cur = build_dspine_write(*this, model, cur, inp_tokens, il, il == n_layer - 1 ? &dspine : nullptr);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    dspine_raw = dspine.raw;
    dspine_rms = dspine.rms;
    dspine_q   = dspine.q;

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (model.dspine_codebook) {
        build_dspine_lattice(*this, model, res->t_inp_tokens, output, dspine_raw, dspine_rms, dspine_q);
    }

    return cur;
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:
                    if (model->dspine_codebook) {
                        return std::make_unique<graph_dflash>(params);
                    }
                    return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
