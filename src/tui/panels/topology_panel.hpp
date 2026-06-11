#pragma once

#include "tracer/topology.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

ftxui::Element render_topology_panel(const trace::Topology& topology,
                                     const AppState& state,
                                     bool focused);

}  // namespace tui
