#include "tui/panels/topology_panel.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace tui {

namespace {

constexpr int kTopologyViewportLines = 14;

struct TopologyLine {
    int node_index = -1;
    ftxui::Element element;
};

static std::vector<TopologyLine> build_topology_lines(const trace::Topology& topology,
                                                      const AppState& state) {
    const auto& nodes = topology.nodes();
    std::vector<TopologyLine> lines;

    lines.push_back({-1, ftxui::text("▼ " + (nodes.empty() ? "model" : nodes[0].label))});

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

        std::string marker = "  ";
        if (n.expandable) {
            marker = n.expanded ? "▼ " : "▶ ";
        } else if (target) {
            marker = "● ";
        }

        std::string line = prefix + marker + n.label;
        if (target) {
            line += "  [Active Capture Target]";
        }
        if (selected) {
            line = "> " + line;
        }

        auto el = ftxui::text(line);
        if (selected) {
            el = el | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
        } else if (target) {
            el = el | ftxui::color(ftxui::Color::Yellow);
        }
        lines.push_back({i, std::move(el)});
    }

    return lines;
}

}  // namespace

int topology_visible_line_of(const trace::Topology& topology, int node_index) {
    const auto& nodes = topology.nodes();
    int line = 1;  // header line
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (nodes[static_cast<size_t>(i)].depth == 0) {
            continue;
        }
        if (i == node_index) {
            return line;
        }
        ++line;
    }
    return 1;
}

int topology_next_node(const trace::Topology& topology, int node_index, int delta) {
    const auto& nodes = topology.nodes();
    const int n = static_cast<int>(nodes.size());
    if (n == 0) {
        return 0;
    }

    int i = node_index;
    while (true) {
        i += delta;
        if (i < 0 || i >= n) {
            return node_index;
        }
        if (nodes[static_cast<size_t>(i)].depth > 0) {
            return i;
        }
    }
}

int topology_first_node(const trace::Topology& topology) {
    const auto& nodes = topology.nodes();
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (nodes[static_cast<size_t>(i)].depth > 0) {
            return i;
        }
    }
    return 0;
}

int topology_last_node(const trace::Topology& topology) {
    const auto& nodes = topology.nodes();
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        if (nodes[static_cast<size_t>(i)].depth > 0) {
            return i;
        }
    }
    return 0;
}

void topology_ensure_visible(const trace::Topology& topology, AppState& state) {
    const int line = topology_visible_line_of(topology, state.topology_cursor);
    if (line < state.topology_scroll + 1) {
        state.topology_scroll = std::max(0, line - 1);
    } else if (line >= state.topology_scroll + kTopologyViewportLines) {
        state.topology_scroll = line - kTopologyViewportLines + 1;
    }
}

ftxui::Element render_topology_panel(const trace::Topology& topology,
                                     const AppState& state,
                                     bool focused) {
    const auto lines = build_topology_lines(topology, state);
    const int scroll = std::clamp(state.topology_scroll, 0,
                                  std::max(0, static_cast<int>(lines.size()) - 1));

    std::vector<ftxui::Element> visible;
    const int end = std::min(static_cast<int>(lines.size()), scroll + kTopologyViewportLines);
    for (int i = scroll; i < end; ++i) {
        visible.push_back(lines[static_cast<size_t>(i)].element);
    }

    if (static_cast<int>(lines.size()) > kTopologyViewportLines) {
        visible.push_back(ftxui::text("j/k: move  g/G: top/bottom  space/r: capture  (" +
                                      std::to_string(state.topology_cursor) + "/" +
                                      std::to_string(topology.nodes().size() - 1) + ")") |
                          ftxui::dim | ftxui::color(ftxui::Color::GrayLight));
    }

    auto body = ftxui::vbox(visible);
    auto title = focused ? "1. MODEL TOPOLOGY (Focus Active)" : "1. MODEL TOPOLOGY";
    return ftxui::window(ftxui::text(title), body) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
