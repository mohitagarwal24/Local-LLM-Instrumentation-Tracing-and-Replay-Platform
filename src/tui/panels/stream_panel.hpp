#pragma once

#include "tracer/event.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>
#include <vector>

namespace tui {

ftxui::Element render_stream_panel(const std::vector<trace::TraceEvent>& events,
                                   const AppState& state,
                                   bool focused);

}  // namespace tui
