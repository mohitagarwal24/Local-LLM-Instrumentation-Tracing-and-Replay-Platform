#include "tui/panels/attention_panel.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace tui {

static char heat_char(float v, float contrast) {
    const float x = std::clamp(v * contrast, 0.f, 1.f);
    if (x > 0.85f) {
        return static_cast<char>(0xDB);
    }
    if (x > 0.65f) {
        return '%';
    }
    if (x > 0.45f) {
        return '+';
    }
    if (x > 0.25f) {
        return '.';
    }
    if (x > 0.05f) {
        return '-';
    }
    return ' ';
}

ftxui::Element render_attention_panel(const trace::AttentionSnapshot& attention,
                                      const AppState& state,
                                      bool focused) {
    const int n = attention.n_tokens > 0 ? attention.n_tokens : 8;
    const int window = std::min(8, n);

    const int max_head = attention.n_heads > 0 ? attention.n_heads - 1 : 0;
    const int head = std::clamp(state.attn_head, 0, max_head);

    std::ostringstream header;
    header << "Tokens: viewport [" << state.attn_pan_x << "-" << (state.attn_pan_x + window)
           << "] x [" << state.attn_pan_y << "-" << (state.attn_pan_y + window)
           << "]  Head " << head << "/" << max_head;

    std::vector<ftxui::Element> rows;
    rows.push_back(ftxui::text(header.str()));
    rows.push_back(ftxui::separator());

    for (int r = state.attn_pan_y; r < std::min(n, state.attn_pan_y + window); ++r) {
        std::string line;
        for (int c = state.attn_pan_x; c < std::min(n, state.attn_pan_x + window); ++c) {
            float v = 0.f;
            if (!attention.weights.empty() && r < attention.n_tokens && c < attention.n_tokens) {
                v = attention.weight_at(head, r, c);
            } else {
                v = (r == c) ? 1.f : 0.05f * std::exp(-0.2f * std::abs(r - c));
            }
            line += heat_char(v, state.attn_contrast);
            line += heat_char(v, state.attn_contrast);
            line += ' ';
        }
        rows.push_back(ftxui::text(line) | ftxui::color(ftxui::Color::GreenLight));
    }

    rows.push_back(ftxui::separator());
    rows.push_back(ftxui::text("[hjkl]: Pan  [+/-]: Contrast  [[]/[]]: Switch head (instant)  [Tab]: Focus"));

    auto title = focused ? "3. ATTENTION MATRIX (Focus Active)" : "3. ATTENTION MATRIX";
    return ftxui::window(ftxui::text(title), ftxui::vbox(rows)) |
           (focused ? ftxui::borderDouble : ftxui::border);
}

}  // namespace tui
