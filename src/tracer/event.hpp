#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace trace {

enum class LayerType : uint8_t {
    Unknown = 0,
    Embed,
    Attention,
    MLP,
    Norm,
    Output,
};

enum class AnomalyKind : uint8_t {
    None = 0,
    NaN,
    Inf,
    OutlierMax,
    HighSparsity,
    LowSparsity,
};

struct TensorStats {
    float mean   = 0.f;
    float min    = 0.f;
    float max    = 0.f;
    float sparsity = 0.f;  // fraction of values near zero
    bool  has_nan = false;
    bool  has_inf = false;
};

struct TraceEvent {
    uint64_t id = 0;
    std::chrono::steady_clock::time_point timestamp{};
    std::string tensor_name;
    std::string op_name;
    LayerType layer_type = LayerType::Unknown;
    int layer_index = -1;

    std::array<int64_t, 4> shape{{0, 0, 0, 0}};
    int ggml_type = 0;
    std::string device;

    TensorStats stats{};
    double latency_ms = 0.0;
    double block_latency_ms = 0.0;
};

struct AttentionSnapshot {
    int layer = 0;
    int n_heads = 0;
    int n_tokens = 0;
    // row-major per head: weights[head * n_tokens * n_tokens + row * n_tokens + col]
    std::vector<float> weights;
    std::chrono::steady_clock::time_point timestamp{};

    float weight_at(int head, int row, int col) const {
        if (n_tokens <= 0 || n_heads <= 0 || weights.empty()) {
            return 0.f;
        }
        const int h = std::max(0, std::min(head, n_heads - 1));
        const size_t idx = static_cast<size_t>(h * n_tokens * n_tokens + row * n_tokens + col);
        return idx < weights.size() ? weights[idx] : 0.f;
    }
};

struct AnomalyRecord {
    std::chrono::steady_clock::time_point timestamp{};
    AnomalyKind kind = AnomalyKind::None;
    std::string message;
    int layer_index = -1;
};

inline const char* layer_type_name(LayerType t) {
    switch (t) {
    case LayerType::Embed:      return "Embed";
    case LayerType::Attention:  return "Attn (Self)";
    case LayerType::MLP:        return "MLP (SwiGLU)";
    case LayerType::Norm:       return "LayerNorm";
    case LayerType::Output:     return "Output";
    default:                    return "Unknown";
    }
}

inline const char* anomaly_kind_symbol(AnomalyKind k) {
    switch (k) {
    case AnomalyKind::NaN:          return "✖";
    case AnomalyKind::Inf:          return "✖";
    case AnomalyKind::OutlierMax:   return "⚠";
    case AnomalyKind::HighSparsity: return "⚠";
    case AnomalyKind::LowSparsity:  return "ℹ";
    default:                        return "·";
    }
}

}  // namespace trace
