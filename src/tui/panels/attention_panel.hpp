#pragma once

#include "tracer/event.hpp"
#include "tui/app.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

ftxui::Element render_attention_panel(const trace::AttentionSnapshot& attention,
                                      const AppState& state,
                                      bool focused);

}  // namespace tui
