#pragma once

#include "tracer/topology.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

ftxui::Element render_topology_panel(const trace::Topology& topology,
                                     const AppState& state,
                                     bool focused);

int topology_next_node(const trace::Topology& topology, int node_index, int delta);
int topology_first_node(const trace::Topology& topology);
int topology_last_node(const trace::Topology& topology);
void topology_ensure_visible(const trace::Topology& topology, AppState& state);

}  // namespace tui
