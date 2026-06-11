#include "tui/panels/anomaly_panel.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace tui {

static std::string short_time(const std::chrono::steady_clock::time_point& tp) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    const auto sec = duration_cast<seconds>(tp.time_since_epoch()) % 60;
    const auto min = duration_cast<minutes>(tp.time_since_epoch()) % 60;
    const auto hr = duration_cast<hours>(tp.time_since_epoch()) % 24;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hr.count() << ':' << std::setw(2) << min.count()
        << ':' << std::setw(2) << sec.count() << '.' << std::setw(3) << ms.count();
    return oss.str();
}

ftxui::Element render_anomaly_panel(const std::vector<trace::AnomalyRecord>& anomalies,
                                    const AppState& state,
                                    bool focused) {
    std::vector<ftxui::Element> lines;
    const int start = std::max(0, static_cast<int>(anomalies.size()) - 6 - state.anomaly_scroll);
    const int end = std::min(static_cast<int>(anomalies.size()), start + 6);
    if (start >= end) {
        lines.push_back(ftxui::text("No anomalies recorded"));
    }
    for (int i = start; i < end; ++i) {
        const auto& a = anomalies[static_cast<size_t>(i)];
        std::ostringstream oss;
        oss << short_time(a.timestamp) << ' ' << trace::anomaly_kind_symbol(a.kind) << ' '
            << a.message;
        auto color = ftxui::Color::Yellow;
        if (a.kind == trace::AnomalyKind::NaN || a.kind == trace::AnomalyKind::Inf) {
            color = ftxui::Color::Red;
        }
        lines.push_back(ftxui::text(oss.str()) | ftxui::color(color));
    }

    auto title = focused ? "5. NUMERICAL ANOMALY LEDGER (Focus Active)"
                         : "5. NUMERICAL ANOMALY LEDGER";
    return ftxui::window(ftxui::text(title), ftxui::vbox(lines)) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
