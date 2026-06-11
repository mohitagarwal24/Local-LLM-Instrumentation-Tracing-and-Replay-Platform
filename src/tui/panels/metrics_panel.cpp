#include "tui/panels/metrics_panel.hpp"

#include "ggml.h"

#include <sstream>
#include <string>

namespace tui {

static std::string sparsity_bar(float sparsity) {
    const int filled = static_cast<int>(sparsity * 10.f);
    std::string bar;
    for (int i = 0; i < 10; ++i) {
        bar += (i < filled) ? '#' : '.';
    }
    return bar;
}

ftxui::Element render_metrics_panel(const std::optional<trace::TraceEvent>& selected,
                                    bool focused) {
    std::vector<ftxui::Element> lines;
    if (!selected) {
        lines.push_back(ftxui::text("No event selected"));
    } else {
        const auto& e = *selected;
        std::ostringstream shape;
        shape << "Tensor Shape : [" << e.shape[0] << ", " << e.shape[1] << ", " << e.shape[2]
              << ", " << e.shape[3] << "]";
        lines.push_back(ftxui::text(shape.str()));
        lines.push_back(ftxui::text("Dtype: " + std::string(ggml_type_name(static_cast<ggml_type>(e.ggml_type)))));
        std::ostringstream sp;
        sp << "Sparsity Rate: " << sparsity_bar(e.stats.sparsity) << " "
           << static_cast<int>(e.stats.sparsity * 100.f) << "%";
        lines.push_back(ftxui::text(sp.str()));
        std::ostringstream lat;
        lat << "Latency Delta: " << e.latency_ms << " ms";
        if (e.latency_ms < 5.0) {
            lat << " (Within Normal Bounds)";
        } else {
            lat << " (Elevated)";
        }
        lines.push_back(ftxui::text(lat.str()));
        lines.push_back(ftxui::text("Block Latency: " + std::to_string(e.block_latency_ms) + " ms"));
        lines.push_back(ftxui::text("Mean/Min/Max: " + std::to_string(e.stats.mean) + " / " +
                                    std::to_string(e.stats.min) + " / " + std::to_string(e.stats.max)));
    }

    auto title = focused ? "4. RUNTIME METRICS (Focus Active)" : "4. RUNTIME METRICS";
    return ftxui::window(ftxui::text(title), ftxui::vbox(lines)) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
