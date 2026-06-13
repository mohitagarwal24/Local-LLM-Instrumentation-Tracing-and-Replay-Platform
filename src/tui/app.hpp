#pragma once

#include "engine/llama_runner.hpp"
#include "tracer/recorder.hpp"
#include "tracer/tracer.hpp"
#include "tracer/topology.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace tui {

enum class PanelFocus {
    Topology = 0,
    PacketStream,
    Attention,
    Metrics,
    AnomalyLedger,
    Count
};

struct AppState {
    PanelFocus focus = PanelFocus::Topology;
    int topology_cursor = 1;   // skip root node (not shown in list)
    int topology_scroll = 0;   // first visible line in topology panel
    int packet_scroll = 0;
    int anomaly_scroll = 0;
    int attn_head = 0;
    int attn_pan_x = 0;
    int attn_pan_y = 0;
    float attn_contrast = 1.0f;
    bool quit = false;
};

class TraceApp {
public:
  TraceApp(engine::LlamaRunner* runner, trace::Replayer* replayer, bool record_trace,
           const std::string& record_path);

    int run();

private:
    void pump_events();
    void update_selected_layer();
    void trigger_capture_pass();

    engine::LlamaRunner* runner_;
    trace::Replayer* replayer_;
    bool record_trace_ = false;
    std::unique_ptr<trace::Recorder> recorder_;
    AppState state_{};
    std::vector<trace::TraceEvent> cached_events_;
    std::vector<trace::AnomalyRecord> cached_anomalies_;
    trace::AttentionSnapshot attention_{};
    trace::Topology replay_topology_{};
};

}  // namespace tui
