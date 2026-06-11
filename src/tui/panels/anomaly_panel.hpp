#pragma once

#include "tracer/event.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>
#include <vector>

namespace tui {

ftxui::Element render_anomaly_panel(const std::vector<trace::AnomalyRecord>& anomalies,
                                    const AppState& state,
                                    bool focused);

}  // namespace tui
