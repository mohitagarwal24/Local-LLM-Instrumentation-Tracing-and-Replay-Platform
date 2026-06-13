#include "tui/app.hpp"

#include "tui/panels/anomaly_panel.hpp"
#include "tui/panels/attention_panel.hpp"
#include "tui/panels/metrics_panel.hpp"
#include "tui/panels/stream_panel.hpp"
#include "tui/panels/topology_panel.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>

namespace tui {

TraceApp::TraceApp(engine::LlamaRunner* runner, trace::Replayer* replayer, bool record_trace,
                   const std::string& record_path)
    : runner_(runner), replayer_(replayer), record_trace_(record_trace) {
    if (record_trace_) {
        recorder_ = std::make_unique<trace::Recorder>(record_path);
        recorder_->open();
    }
    if (replayer_) {
        replayer_->load();
        cached_events_ = replayer_->events();
        cached_anomalies_ = replayer_->anomalies();
        if (!replayer_->attentions().empty()) {
            attention_ = replayer_->attentions().back();
        }
        replay_topology_.build_from_model(nullptr, "replay-session");
    }
}

void TraceApp::pump_events() {
    if (!runner_) {
        return;
    }

    while (auto ev = runner_->tracer().events().pop()) {
        cached_events_.push_back(*ev);
        if (recorder_ && recorder_->is_open()) {
            recorder_->write_event(*ev);
        }
    }
    while (auto rec = runner_->tracer().anomalies().pop()) {
        cached_anomalies_.push_back(*rec);
        if (recorder_ && recorder_->is_open()) {
            recorder_->write_anomaly(*rec);
        }
    }

    const auto attn = runner_->tracer().attention_copy();
    if (!attn.weights.empty()) {
        attention_ = attn;
        if (attention_.n_heads > 0) {
            state_.attn_head = std::clamp(state_.attn_head, 0, attention_.n_heads - 1);
        }
        if (recorder_ && recorder_->is_open()) {
            recorder_->write_attention(attention_);
        }
    }

    if (cached_events_.size() > 1024) {
        cached_events_.erase(cached_events_.begin(),
                             cached_events_.begin() + static_cast<long>(cached_events_.size() - 1024));
    }
}

void TraceApp::update_selected_layer() {
    if (!runner_) {
        return;
    }
    const auto& nodes = runner_->topology().nodes();
    if (state_.topology_cursor < 0 || state_.topology_cursor >= static_cast<int>(nodes.size())) {
        return;
    }
    const int layer = nodes[static_cast<size_t>(state_.topology_cursor)].layer_index;
    if (layer >= 0) {
        runner_->tracer().set_capture_target(layer);
    }
}

void TraceApp::trigger_capture_pass() {
    if (!runner_) {
        return;
    }
    update_selected_layer();
    cached_events_.clear();
    cached_anomalies_.clear();
    attention_ = {};
    state_.attn_pan_x = 0;
    state_.attn_pan_y = 0;
    runner_->request_decode();
}

int TraceApp::run() {
    auto screen = ftxui::ScreenInteractive::Fullscreen();

    if (runner_) {
        runner_->start_worker();
        runner_->request_decode();
    }

    auto component = ftxui::CatchEvent(ftxui::Renderer([&] {
        pump_events();

        const bool f_topo = state_.focus == PanelFocus::Topology;
        const bool f_stream = state_.focus == PanelFocus::PacketStream;
        const bool f_attn = state_.focus == PanelFocus::Attention;
        const bool f_metrics = state_.focus == PanelFocus::Metrics;
        const bool f_anom = state_.focus == PanelFocus::AnomalyLedger;

        std::optional<trace::TraceEvent> selected;
        if (!cached_events_.empty()) {
            selected = cached_events_.back();
        }

        const trace::Topology& topo =
            runner_ ? runner_->topology() : replay_topology_;

        std::string status;
        if (runner_ && runner_->running()) {
            status = "  |  [Capturing...]";
        }
        auto help = ftxui::text("[Tab]: Focus  |  [Q]: Quit  |  Topology: j/k, g/G, [Space]/[r]: capture" +
                                status) |
                    ftxui::center;

        auto top_row = ftxui::hbox({
            render_topology_panel(topo, state_, f_topo) | ftxui::flex,
            render_stream_panel(cached_events_, state_, f_stream) | ftxui::flex,
        });

        auto attn = render_attention_panel(attention_, state_, f_attn);
        auto bottom = ftxui::hbox({
            render_metrics_panel(selected, f_metrics) | ftxui::flex,
            render_anomaly_panel(cached_anomalies_, state_, f_anom) | ftxui::flex,
        });

        return ftxui::vbox({help, top_row | ftxui::flex, attn, bottom}) | ftxui::flex;
    }), [&](const ftxui::Event& event) {
        if (event == ftxui::Event::Character('q') || event == ftxui::Event::Character('Q')) {
            state_.quit = true;
            screen.ExitLoopClosure()();
            return true;
        }
        if (event == ftxui::Event::Tab) {
            state_.focus = static_cast<PanelFocus>(
                (static_cast<int>(state_.focus) + 1) % static_cast<int>(PanelFocus::Count));
            return true;
        }

        if (state_.focus == PanelFocus::Topology) {
            const trace::Topology& topo =
                runner_ ? runner_->topology() : replay_topology_;

            const auto move_topology = [&](int delta) {
                state_.topology_cursor = topology_next_node(topo, state_.topology_cursor, delta);
                topology_ensure_visible(topo, state_);
            };

            if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                move_topology(+1);
                return true;
            }
            if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                move_topology(-1);
                return true;
            }
            if (event == ftxui::Event::Character('g')) {
                state_.topology_cursor = topology_first_node(topo);
                topology_ensure_visible(topo, state_);
                return true;
            }
            if (event == ftxui::Event::Character('G')) {
                state_.topology_cursor = topology_last_node(topo);
                topology_ensure_visible(topo, state_);
                return true;
            }
            if (event == ftxui::Event::Character(' ') || event == ftxui::Event::Character('r')) {
                if (runner_) {
                    runner_->topology().set_capture_target(state_.topology_cursor);
                    trigger_capture_pass();
                } else {
                    replay_topology_.set_capture_target(state_.topology_cursor);
                }
                return true;
            }
        }

        if (state_.focus == PanelFocus::PacketStream) {
            if (event == ftxui::Event::Character('j')) {
                state_.packet_scroll = std::max(0, state_.packet_scroll - 1);
                return true;
            }
            if (event == ftxui::Event::Character('k')) {
                state_.packet_scroll += 1;
                return true;
            }
        }

        if (state_.focus == PanelFocus::Attention) {
            if (event == ftxui::Event::Character('h')) {
                state_.attn_pan_x = std::max(0, state_.attn_pan_x - 1);
                return true;
            }
            if (event == ftxui::Event::Character('l')) {
                state_.attn_pan_x += 1;
                return true;
            }
            if (event == ftxui::Event::Character('j')) {
                state_.attn_pan_y += 1;
                return true;
            }
            if (event == ftxui::Event::Character('k')) {
                state_.attn_pan_y = std::max(0, state_.attn_pan_y - 1);
                return true;
            }
            if (event == ftxui::Event::Character('+') || event == ftxui::Event::Character('=')) {
                state_.attn_contrast = std::min(4.f, state_.attn_contrast + 0.25f);
                return true;
            }
            if (event == ftxui::Event::Character('-')) {
                state_.attn_contrast = std::max(0.25f, state_.attn_contrast - 0.25f);
                return true;
            }
            if (event == ftxui::Event::Character('[')) {
                if (state_.attn_head <= 0) {
                    return true;
                }
                state_.attn_head -= 1;
                return true;
            }
            if (event == ftxui::Event::Character(']')) {
                const int max_head = attention_.n_heads > 0
                                         ? attention_.n_heads - 1
                                         : (runner_ ? std::max(0, runner_->n_heads() - 1) : 0);
                if (state_.attn_head >= max_head) {
                    return true;
                }
                state_.attn_head += 1;
                return true;
            }
        }

        if (state_.focus == PanelFocus::AnomalyLedger) {
            if (event == ftxui::Event::Character('j')) {
                state_.anomaly_scroll = std::max(0, state_.anomaly_scroll - 1);
                return true;
            }
            if (event == ftxui::Event::Character('k')) {
                state_.anomaly_scroll += 1;
                return true;
            }
        }

        return false;
    });

    screen.Loop(component);

    if (runner_) {
        runner_->request_stop();
        runner_->wait();
    }
    if (recorder_) {
        recorder_->close();
    }

    return 0;
}

}  // namespace tui
