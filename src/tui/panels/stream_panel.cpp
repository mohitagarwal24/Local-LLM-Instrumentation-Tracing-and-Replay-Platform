#include "tui/panels/stream_panel.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace tui {

static std::string format_time(const std::chrono::steady_clock::time_point& tp) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    const auto sec = duration_cast<seconds>(tp.time_since_epoch()) % 60;
    const auto min = duration_cast<minutes>(tp.time_since_epoch()) % 60;
    const auto hr = duration_cast<hours>(tp.time_since_epoch()) % 24;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hr.count() << ':'
        << std::setw(2) << min.count() << ':' << std::setw(2) << sec.count()
        << '.' << std::setw(3) << ms.count();
    return oss.str();
}

ftxui::Element render_stream_panel(const std::vector<trace::TraceEvent>& events,
                                   const AppState& state,
                                   bool focused) {
    std::vector<ftxui::Element> rows;
    rows.push_back(ftxui::hbox({
        ftxui::text(" ID  ") | ftxui::bold,
        ftxui::text("│ TIMESTAMP    ") | ftxui::bold,
        ftxui::text("│ LAYER TYPE   ") | ftxui::bold,
        ftxui::text("│ COMPUTE DEVICE") | ftxui::bold,
    }));
    rows.push_back(ftxui::separator());

    const int start = std::max(0, static_cast<int>(events.size()) - 12 - state.packet_scroll);
    const int end = std::min(static_cast<int>(events.size()), start + 12);
    for (int i = start; i < end; ++i) {
        const auto& e = events[static_cast<size_t>(i)];
        std::ostringstream id;
        id << std::setw(3) << e.id;
        rows.push_back(ftxui::hbox({
            ftxui::text(" " + id.str() + " ") | ftxui::color(ftxui::Color::GrayLight),
            ftxui::text("│ " + format_time(e.timestamp) + " "),
            ftxui::text("│ " + std::string(trace::layer_type_name(e.layer_type)) + " "),
            ftxui::text("│ " + e.device),
        }));
    }

    auto title = focused ? "2. LIVE PACKET STREAM (Focus Active)" : "2. LIVE PACKET STREAM";
    return ftxui::window(ftxui::text(title), ftxui::vbox(rows)) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
