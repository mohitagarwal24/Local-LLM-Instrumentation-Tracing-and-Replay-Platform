#include "tracer/event.hpp"
#include "tracer/recorder.hpp"

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "demo.trace";

    trace::Recorder rec(path);
    if (!rec.open()) {
        std::fprintf(stderr, "Failed to open %s\n", path.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 1; i <= 48; ++i) {
        trace::TraceEvent ev;
        ev.id = i;
        ev.timestamp = t0 + std::chrono::milliseconds(static_cast<int>(i * 3));
        ev.tensor_name = "layers." + std::to_string((i / 2) % 4) + ".attn";
        ev.op_name = "MUL_MAT";
        ev.layer_type = (i % 3 == 0) ? trace::LayerType::MLP : trace::LayerType::Attention;
        ev.layer_index = static_cast<int>((i / 2) % 4);
        ev.shape = {{32, 32, 4096, 1}};
        ev.ggml_type = 0;
        ev.device = "CPU";
        ev.stats.mean = 0.02f * static_cast<float>(i);
        ev.stats.min = -1.5f;
        ev.stats.max = 2.0f + 0.1f * static_cast<float>(i % 10);
        ev.stats.sparsity = 0.35f + 0.01f * static_cast<float>(i % 20);
        ev.latency_ms = 0.8 + static_cast<double>(i % 7) * 0.15;
        ev.block_latency_ms = 1.0 + static_cast<double>(i % 5) * 0.2;
        rec.write_event(ev);

        if (i % 11 == 0) {
            trace::AnomalyRecord a;
            a.timestamp = ev.timestamp;
            a.kind = trace::AnomalyKind::OutlierMax;
            a.layer_index = ev.layer_index;
            a.message = "Outlier Feature Layer " + std::to_string(ev.layer_index) + ": Max > 6.0";
            rec.write_anomaly(a);
        }
    }

    trace::AttentionSnapshot attn;
    attn.layer = 1;
    attn.head = 0;
    attn.n_tokens = 8;
    attn.timestamp = t0 + std::chrono::milliseconds(150);
    attn.weights.resize(64);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            attn.weights[static_cast<size_t>(r) * 8 + c] =
                (c <= r) ? 0.9f / static_cast<float>(r + 1) : 0.02f;
        }
    }
    rec.write_attention(attn);
    rec.close();

    std::printf("Wrote demo trace to %s\n", path.c_str());
    return 0;
}
