#include "tui/panels/topology_panel.hpp"

#include <string>

namespace tui {

static std::string indent_prefix(int depth, bool last_child) {
    std::string s;
    for (int i = 0; i < depth; ++i) {
        s += (i == depth - 1) ? (last_child ? "  " : "│ ") : "│ ";
    }
    return s;
}

ftxui::Element render_topology_panel(const trace::Topology& topology,
                                     const AppState& state,
                                     bool focused) {
    const auto& nodes = topology.nodes();
    std::vector<ftxui::Element> lines;
    lines.push_back(ftxui::text("▼ " + (nodes.empty() ? "model" : nodes[0].label)));

    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const auto& n = nodes[static_cast<size_t>(i)];
        if (n.depth == 0) {
            continue;
        }

        std::string prefix;
        if (n.depth == 1) {
            prefix = "  ";
        } else if (n.depth == 2) {
            prefix = "   ";
        } else {
            prefix = "     ";
        }

        const bool selected = (i == state.topology_cursor);
        const bool target = n.is_capture_target;

        std::string marker = "► ";
        if (n.expandable) {
            marker = n.expanded ? "▼ " : "▶ ";
        } else if (target) {
            marker = "● ";
        } else {
            marker = "  ";
        }

        std::string line = prefix + marker + n.label;
        if (target) {
            line += "  [Active Capture Target]";
        }

        auto el = ftxui::text(line);
        if (selected) {
            el = el | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
        } else if (target) {
            el = el | ftxui::color(ftxui::Color::Yellow);
        }
        lines.push_back(el);
    }

    auto body = ftxui::vbox(lines);
    auto title = focused ? "1. MODEL TOPOLOGY (Focus Active)" : "1. MODEL TOPOLOGY";
    return ftxui::window(ftxui::text(title), body) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
