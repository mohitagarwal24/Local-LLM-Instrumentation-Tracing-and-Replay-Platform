#pragma once

#include "tracer/event.hpp"
#include "tracer/ring_buffer.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct ggml_tensor;

namespace trace {

struct TracerConfig {
    float outlier_max_threshold = 6.0f;
    float sparsity_epsilon = 1e-6f;
    int   attention_layer = 0;
    int   attention_head = 0;
    bool  capture_attention = true;
    bool  verbose_stdout = false;
};

class Tracer {
public:
    explicit Tracer(TracerConfig config = {});

    void reset();
    void set_capture_target(int layer_index);
    void set_attention_head(int head_index);
    void set_verbose_stdout(bool enabled) { config_.verbose_stdout = enabled; }

    EventRingBuffer& events() { return events_; }
    AnomalyRingBuffer& anomalies() { return anomalies_; }

    const AttentionSnapshot* latest_attention() const;
    AttentionSnapshot attention_copy() const;

    static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data);

private:
    bool on_tensor(struct ggml_tensor* t);
    TensorStats compute_stats(struct ggml_tensor* t, std::vector<uint8_t>& scratch);
    LayerType classify(const std::string& name, int& layer_index) const;
    void check_anomalies(const TraceEvent& ev);
    bool maybe_capture_attention(struct ggml_tensor* t, std::vector<uint8_t>& scratch);

    TracerConfig config_;
    EventRingBuffer events_;
    AnomalyRingBuffer anomalies_;

    std::atomic<uint64_t> next_id_{1};
    std::chrono::steady_clock::time_point block_start_{};
    int current_block_layer_ = -1;

    mutable std::mutex attention_mu_;
    std::optional<AttentionSnapshot> latest_attention_;

    std::vector<uint8_t> host_scratch_;
    std::chrono::steady_clock::time_point last_node_time_{};
};

}  // namespace trace
