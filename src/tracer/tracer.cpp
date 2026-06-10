#include "tracer/tracer.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace trace {

Tracer::Tracer(TracerConfig config) : config_(std::move(config)) {
    reset();
}

void Tracer::reset() {
    events_.clear();
    anomalies_.clear();
    next_id_ = 1;
    block_start_ = std::chrono::steady_clock::now();
    last_node_time_ = block_start_;
    current_block_layer_ = -1;
    std::lock_guard<std::mutex> lock(attention_mu_);
    latest_attention_.reset();
}

void Tracer::set_capture_target(int layer_index) {
    config_.attention_layer = layer_index;
}

void Tracer::set_attention_head(int head_index) {
    config_.attention_head = std::max(0, head_index);
}

const AttentionSnapshot* Tracer::latest_attention() const {
    std::lock_guard<std::mutex> lock(attention_mu_);
    if (!latest_attention_) {
        return nullptr;
    }
    return &*latest_attention_;
}

AttentionSnapshot Tracer::attention_copy() const {
    std::lock_guard<std::mutex> lock(attention_mu_);
    return latest_attention_ ? *latest_attention_ : AttentionSnapshot{};
}

bool Tracer::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* self = static_cast<Tracer*>(user_data);
    if (ask) {
        return true;
    }
    return self->on_tensor(t);
}

static float read_elem(const uint8_t* data, ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:  return *reinterpret_cast<const float*>(data);
    case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
    case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(data));
    case GGML_TYPE_I32:  return static_cast<float>(*reinterpret_cast<const int32_t*>(data));
    case GGML_TYPE_I16:  return static_cast<float>(*reinterpret_cast<const int16_t*>(data));
    case GGML_TYPE_I8:   return static_cast<float>(*reinterpret_cast<const int8_t*>(data));
    default:             return 0.f;
    }
}

static bool is_floatish(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

TensorStats Tracer::compute_stats(struct ggml_tensor* t, std::vector<uint8_t>& scratch) {
    TensorStats stats{};
    if (!t || !t->data || ggml_is_empty(t) || !is_floatish(t->type)) {
        return stats;
    }

    const bool is_host = t->buffer ? ggml_backend_buffer_is_host(t->buffer) : true;
    uint8_t* data = nullptr;
    if (is_host) {
        data = static_cast<uint8_t*>(t->data);
    } else {
        const size_t nbytes = ggml_nbytes(t);
        scratch.resize(nbytes);
        ggml_backend_tensor_get(t, scratch.data(), 0, nbytes);
        data = scratch.data();
    }

    double sum = 0.0;
    uint64_t count = 0;
    uint64_t near_zero = 0;
    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();

    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t i = static_cast<size_t>(i3 * t->nb[3] + i2 * t->nb[2] +
                                                         i1 * t->nb[1] + i0 * t->nb[0]);
                    const float v = read_elem(data + i, t->type);
                    if (std::isnan(v)) {
                        stats.has_nan = true;
                        continue;
                    }
                    if (std::isinf(v)) {
                        stats.has_inf = true;
                        continue;
                    }
                    sum += v;
                    ++count;
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                    if (std::fabs(v) < config_.sparsity_epsilon) {
                        ++near_zero;
                    }
                }
            }
        }
    }

    if (count > 0) {
        stats.mean = static_cast<float>(sum / static_cast<double>(count));
        stats.min = vmin;
        stats.max = vmax;
        stats.sparsity = static_cast<float>(near_zero) / static_cast<float>(count);
    }
    return stats;
}

LayerType Tracer::classify(const std::string& name, int& layer_index) const {
    layer_index = -1;

    if (name.find("embd") != std::string::npos || name.find("token_embd") != std::string::npos) {
        return LayerType::Embed;
    }
    if (name.find("output") != std::string::npos || name.find("lm_head") != std::string::npos) {
        return LayerType::Output;
    }
    if (name.find("norm") != std::string::npos || name.find("rms") != std::string::npos) {
        const auto pos = name.find("layers.");
        if (pos != std::string::npos) {
            layer_index = std::atoi(name.c_str() + pos + 7);
        }
        return LayerType::Norm;
    }
    if (name.find("ffn") != std::string::npos || name.find("mlp") != std::string::npos ||
        name.find("feed_forward") != std::string::npos || name.find("swiglu") != std::string::npos) {
        const auto pos = name.find("layers.");
        if (pos != std::string::npos) {
            layer_index = std::atoi(name.c_str() + pos + 7);
        }
        return LayerType::MLP;
    }
    if (name.find("attn") != std::string::npos || name.find("kq_soft_max") != std::string::npos ||
        name.find("kqv") != std::string::npos || name.find("Qcur") != std::string::npos ||
        name.find("Kcur") != std::string::npos || name.find("Vcur") != std::string::npos) {
        const auto pos = name.find("layers.");
        if (pos != std::string::npos) {
            layer_index = std::atoi(name.c_str() + pos + 7);
        }
        if (layer_index < 0) {
            const auto dash = name.find("kq_soft_max-");
            if (dash != std::string::npos) {
                layer_index = std::atoi(name.c_str() + dash + 12);
            }
        }
        return LayerType::Attention;
    }

    const auto pos = name.find("layers.");
    if (pos != std::string::npos) {
        layer_index = std::atoi(name.c_str() + pos + 7);
    }
    return LayerType::Unknown;
}

void Tracer::check_anomalies(const TraceEvent& ev) {
    auto push = [&](AnomalyKind kind, const std::string& msg) {
        AnomalyRecord rec;
        rec.timestamp = ev.timestamp;
        rec.kind = kind;
        rec.message = msg;
        rec.layer_index = ev.layer_index;
        anomalies_.push(std::move(rec));
    };

    if (ev.stats.has_nan) {
        push(AnomalyKind::NaN, "NaN detected in tensor " + ev.tensor_name);
    }
    if (ev.stats.has_inf) {
        push(AnomalyKind::Inf, "Inf detected in tensor " + ev.tensor_name);
    }
    if (ev.stats.max > config_.outlier_max_threshold) {
        push(AnomalyKind::OutlierMax,
             "Outlier Feature Layer " + std::to_string(ev.layer_index) + ": Max > " +
                 std::to_string(config_.outlier_max_threshold));
    }
    if (ev.stats.sparsity > 0.95f) {
        push(AnomalyKind::HighSparsity,
             "High sparsity (" + std::to_string(static_cast<int>(ev.stats.sparsity * 100)) +
                 "%) in " + ev.tensor_name);
    }
    if (ev.device.rfind("CPU", 0) == 0 && ev.layer_type == LayerType::Norm && ev.latency_ms > 3.0) {
        push(AnomalyKind::OutlierMax,
             "CPU Fallback: elevated latency on " + ev.tensor_name);
    }
}

bool Tracer::maybe_capture_attention(struct ggml_tensor* t, std::vector<uint8_t>& scratch) {
    if (!config_.capture_attention || !t || !t->name) {
        return false;
    }

    const std::string name(t->name);
    if (name.rfind("kq_soft_max", 0) != 0) {
        return false;
    }

    int layer = -1;
    const auto dash = name.find('-');
    if (dash != std::string::npos) {
        layer = std::atoi(name.c_str() + dash + 1);
    }
    if (layer != config_.attention_layer) {
        return false;
    }

    if (!is_floatish(t->type) || t->ne[0] == 0 || t->ne[1] == 0) {
        return false;
    }

    const int n_head = static_cast<int>(t->ne[2] > 0 ? t->ne[2] : 1);
    const int head = std::clamp(config_.attention_head, 0, std::max(0, n_head - 1));
    const int n_tokens = static_cast<int>(t->ne[1]);

    const bool is_host = t->buffer ? ggml_backend_buffer_is_host(t->buffer) : true;
    uint8_t* data = nullptr;
    if (is_host) {
        data = static_cast<uint8_t*>(t->data);
    } else {
        const size_t nbytes = ggml_nbytes(t);
        scratch.resize(nbytes);
        ggml_backend_tensor_get(t, scratch.data(), 0, nbytes);
        data = scratch.data();
    }

    AttentionSnapshot snap;
    snap.layer = layer;
    snap.head = head;
    snap.n_tokens = n_tokens;
    snap.timestamp = std::chrono::steady_clock::now();
    snap.weights.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_tokens), 0.f);

    for (int row = 0; row < n_tokens; ++row) {
        for (int col = 0; col < n_tokens; ++col) {
            const size_t i = static_cast<size_t>(col * t->nb[0] + row * t->nb[1] +
                                                 head * t->nb[2]);
            snap.weights[static_cast<size_t>(row) * n_tokens + col] =
                read_elem(data + i, t->type);
        }
    }

    std::lock_guard<std::mutex> lock(attention_mu_);
    latest_attention_ = std::move(snap);
    return true;
}

bool Tracer::on_tensor(struct ggml_tensor* t) {
    if (!t) {
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    const double latency_ms =
        std::chrono::duration<double, std::milli>(now - last_node_time_).count();
    last_node_time_ = now;

    TraceEvent ev;
    ev.id = next_id_++;
    ev.timestamp = now;
    ev.tensor_name = t->name ? t->name : "";
    ev.op_name = ggml_op_desc(t);
    ev.layer_type = classify(ev.tensor_name, ev.layer_index);
    ev.shape = {{t->ne[0], t->ne[1], t->ne[2], t->ne[3]}};
    ev.ggml_type = static_cast<int>(t->type);
    ev.device = (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) ? "CUDA [GPU 0]"
                                                                       : "CPU";
    ev.latency_ms = latency_ms;

    if (ev.layer_index >= 0 && ev.layer_index != current_block_layer_) {
        current_block_layer_ = ev.layer_index;
        block_start_ = now;
    }
    ev.block_latency_ms =
        std::chrono::duration<double, std::milli>(now - block_start_).count();

    ev.stats = compute_stats(t, host_scratch_);
    check_anomalies(ev);
    maybe_capture_attention(t, host_scratch_);

    events_.push(std::move(ev));

    if (config_.verbose_stdout) {
        const auto& e = events_.snapshot().back();
        std::printf("[%llu] %s | %s | shape=[%lld,%lld,%lld,%lld] mean=%.4f max=%.4f sparsity=%.1f%% latency=%.3fms\n",
                    static_cast<unsigned long long>(e.id),
                    e.tensor_name.c_str(),
                    layer_type_name(e.layer_type),
                    static_cast<long long>(e.shape[0]),
                    static_cast<long long>(e.shape[1]),
                    static_cast<long long>(e.shape[2]),
                    static_cast<long long>(e.shape[3]),
                    e.stats.mean,
                    e.stats.max,
                    e.stats.sparsity * 100.f,
                    e.latency_ms);
    }

    return true;
}

}  // namespace trace
