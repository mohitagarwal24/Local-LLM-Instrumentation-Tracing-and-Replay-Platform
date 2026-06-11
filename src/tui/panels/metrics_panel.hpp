#pragma once

#include "tracer/event.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>
#include <optional>

namespace tui {

ftxui::Element render_metrics_panel(const std::optional<trace::TraceEvent>& selected,
                                    bool focused);

}  // namespace tui
