#pragma once
// Model and runtime metadata for GET /v1/models, GET /v1/models/NAME, GET /props and GET /health,
// published under the field names llama.cpp uses. Clients size their context from the server instead
// of a hand-written config number: AZ PAL reads data[].meta.n_ctx and, failing that, max_context on
// /health or /metrics at the server root. Without either, a profile silently carries whatever number
// whoever copied it last typed in, from a machine with a different context.
//
// Reads the GGUF header once at startup. Nothing here reaches into the engine, so upstream merges
// stay clean: the only shared file this touches is the routing in tqserve.cpp.
#include "json_out.hpp"

#include <src/gguf.hpp>

#include <sys/stat.h>

#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tqserve {

struct ModelInfo {
    std::string id, path, arch, pretty, parent, quantized_by, repo_url, size_label, ftype;
    uint64_t created = 0, size = 0, n_params = 0, n_vocab = 0;
    uint32_t vocab_type = 0;
    uint32_t n_ctx = 0;  // positions ONE request gets: the memory fit may have cut the requested context
    uint32_t n_ctx_train = 0, n_embd = 0, n_layer = 0;
    long long bos = -1, eos = -1;  // -1: the checkpoint does not declare one
    std::vector<std::pair<std::string, uint64_t>> weight_types;  // format name -> tensors, e.g. TQ4: 400
    // tqllm runtime facts, kept in their own JSON object so no client can read them as llama.cpp fields
    uint32_t slots = 1, max_tokens = 0;
    std::string prompt_cache = "off", cache_active = "off", reasoning = "off", effort, kv, kv_store, draft;
};

inline std::string gstr(const tqllm::GgufFile & f, const char * key) {
    return f.has(key) ? f.str(key) : std::string();
}

// Names for the tensor types this reader knows. Formats above ggml id 43 have no llama.cpp
// quantisation name at all, so reporting one would be a fabrication; TQ formats get the name tqllm uses.
inline const char * weight_format_name(uint32_t type) {
    switch (type) {
        case tqllm::GGML_F32:    return "F32";
        case tqllm::GGML_F16:    return "F16";
        case tqllm::GGML_BF16:   return "BF16";
        case tqllm::GGML_TQ3_4S: return "TQ3_4S";
        case tqllm::GGML_TQ4:    return "TQ4";
        case tqllm::GGML_TQ3:    return "TQ3";
        default: return nullptr;  // a standard K-quant / IQ type: llama.cpp names it, we do not guess
    }
}

inline ModelInfo read_model_info(const std::string & path, const std::string & id) {
    ModelInfo m;
    m.id = id;
    m.path = path;
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        m.created = uint64_t(st.st_mtime);
        m.size = uint64_t(st.st_size);
    }
    tqllm::GgufFile f;
    try {
        f = tqllm::gguf_open(path);
    } catch (...) {
        return m;  // a header this reader cannot parse costs the metadata, never the endpoint
    }
    m.arch = gstr(f, "general.architecture");
    m.pretty = gstr(f, "general.name");
    m.parent = gstr(f, "general.base_model.0.name");
    m.quantized_by = gstr(f, "general.quantized_by");
    m.repo_url = gstr(f, "general.repo_url");
    m.size_label = gstr(f, "general.size_label");
    const std::string a = m.arch + ".";
    m.n_ctx_train = uint32_t(f.num(a + "context_length", 0));
    m.n_embd = uint32_t(f.num(a + "embedding_length", 0));
    m.n_layer = uint32_t(f.num(a + "block_count", 0));
    if (const auto it = f.kv.find("tokenizer.ggml.tokens"); it != f.kv.end())
        m.n_vocab = uint64_t(it->second.strs.size());
    m.vocab_type = gstr(f, "tokenizer.ggml.model") == "gpt2" ? 2 : 0;  // llama.cpp reports 2 for the Qwen BPE vocab
    if (f.has("tokenizer.ggml.bos_token_id")) m.bos = static_cast<long long>(f.num("tokenizer.ggml.bos_token_id"));
    if (f.has("tokenizer.ggml.eos_token_id")) m.eos = static_cast<long long>(f.num("tokenizer.ggml.eos_token_id"));

    // n_params is the element count of the tensor directory, the same sum llama.cpp reports. Bits per
    // weight comes from the bytes that actually store the quantised body, NOT from general.file_type:
    // the TQ converter stamps 15 (Q4_K_M) on files whose tensors are TQ4, and embeddings held in F32
    // would dominate any largest-tensor-type answer, so the label only ever names a TQ format.
    std::map<std::string, uint64_t> tensors;
    std::map<std::string, std::pair<uint64_t, uint64_t>> quant;  // TQ name -> bytes, elements
    for (const auto & entry : f.tensors) {
        const tqllm::GgufTensor & t = entry.second;
        uint64_t elems = 1;
        for (const uint64_t d : t.ne) if (d) elems *= d;
        m.n_params += elems;
        if (const char * n = weight_format_name(t.type)) {
            tensors[n]++;
            if (t.type >= tqllm::GGML_TQ3_4S) quant[n].first += t.bytes, quant[n].second += elems;
        }
    }
    for (const auto & e : tensors) m.weight_types.emplace_back(e.first, e.second);
    uint64_t best_bytes = 0, best_elems = 0;
    std::string best;
    for (const auto & e : quant)
        if (e.second.first > best_bytes) best_bytes = e.second.first, best_elems = e.second.second, best = e.first;
    if (!best.empty() && best_elems) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s - %.2f bpw", best.c_str(), double(best_bytes) * 8.0 / double(best_elems));
        m.ftype = buf;
    }
    return m;
}

inline std::string models_json(const ModelInfo & m) {
    // llama.cpp answers /v1/models with the OMA models list AND the OpenAI data list in one body;
    // clients read whichever they know. AZ PAL reads data[].meta.n_ctx.
    std::string o = "{\"models\":[{\"name\":";
    jstr(o, m.id);
    o += ",\"model\":";
    jstr(o, m.path);
    o += ",\"type\":\"model\",\"details\":{\"format\":\"gguf\",\"parent_model\":";
    jstr(o, m.parent);
    o += ",\"parameter_size\":";
    jstr(o, m.size_label);
    o += ",\"quantization_level\":";
    jstr(o, m.ftype);
    o += "}}],\"object\":\"list\",\"data\":[{\"id\":";
    jstr(o, m.id);
    o += ",\"aliases\":[";
    jstr(o, m.id);
    o += "],\"tags\":[],\"object\":\"model\",\"architecture\":{\"input_modalities\":[\"text\"],"
         "\"output_modalities\":[\"text\"]},\"created\":";
    o += std::to_string(m.created);
    o += ",\"owned_by\":\"tqllm\",\"meta\":{\"vocab_type\":" + std::to_string(m.vocab_type);
    o += ",\"n_vocab\":" + std::to_string(m.n_vocab);
    o += ",\"n_ctx\":" + std::to_string(m.n_ctx);
    o += ",\"n_ctx_train\":" + std::to_string(m.n_ctx_train);
    o += ",\"n_embd\":" + std::to_string(m.n_embd);
    o += ",\"n_params\":" + std::to_string(m.n_params);
    o += ",\"size\":" + std::to_string(m.size);
    if (!m.ftype.empty()) {
        o += ",\"ftype\":";
        jstr(o, m.ftype);
    }
    o += "},\"tqllm\":{\"architecture\":";
    jstr(o, m.arch);
    o += ",\"name\":";
    jstr(o, m.pretty);
    o += ",\"quantized_by\":";
    jstr(o, m.quantized_by);
    o += ",\"repo_url\":";
    jstr(o, m.repo_url);
    o += ",\"n_layer\":" + std::to_string(m.n_layer);
    o += ",\"slots\":" + std::to_string(m.slots);
    o += ",\"max_tokens\":" + std::to_string(m.max_tokens);
    o += ",\"kv\":";
    jstr(o, m.kv);
    o += ",\"kv_store\":";
    jstr(o, m.kv_store);
    o += ",\"draft\":";
    jstr(o, m.draft);
    o += ",\"prompt_cache\":";
    jstr(o, m.prompt_cache);
    o += ",\"prompt_cache_active\":";
    jstr(o, m.cache_active);
    o += ",\"reasoning\":";
    jstr(o, m.reasoning);
    o += ",\"reasoning_effort\":";
    jstr(o, m.effort);
    o += ",\"weight_types\":{";
    for (size_t i = 0; i < m.weight_types.size(); ++i) {
        if (i) o += ',';
        jstr(o, m.weight_types[i].first);
        o += ':' + std::to_string(m.weight_types[i].second);
    }
    o += "}}}]}";
    return o;
}

// GET /props. Only keys this server can actually answer are emitted: add_bos_token is absent from this
// checkpoint metadata and the general.sampling.* defaults are not applied by the engine, so claiming
// either would be worse than leaving it out.
inline std::string props_json(const ModelInfo & m) {
    std::string o = "{\"model\":";
    jstr(o, m.id);
    o += ",\"modelpath\":";
    jstr(o, m.path);
    o += ",\"n_ctx_train\":" + std::to_string(m.n_ctx_train);
    o += ",\"vocab\":{\"n_vocab\":" + std::to_string(m.n_vocab);
    o += ",\"vocab_type\":" + std::to_string(m.vocab_type);
    if (m.bos >= 0) o += ",\"bos_token_id\":" + std::to_string(m.bos);
    if (m.eos >= 0) o += ",\"eos_token_id\":" + std::to_string(m.eos) + ",\"eot_token_id\":" + std::to_string(m.eos);
    o += "},\"tqllm\":{\"n_ctx\":" + std::to_string(m.n_ctx);
    o += ",\"slots\":" + std::to_string(m.slots);
    o += ",\"max_tokens\":" + std::to_string(m.max_tokens);
    o += ",\"prompt_cache\":";
    jstr(o, m.prompt_cache);
    o += ",\"prompt_cache_active\":";
    jstr(o, m.cache_active);
    o += ",\"reasoning\":";
    jstr(o, m.reasoning);
    o += ",\"reasoning_effort\":";
    jstr(o, m.effort);
    o += "}}";
    return o;
}

}  // namespace tqserve
