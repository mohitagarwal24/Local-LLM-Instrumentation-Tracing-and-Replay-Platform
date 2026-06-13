# LLM Trace Platform

Non-invasive telemetry, tracing, and replay for local transformer models. Works with any GGUF model loaded through [llama.cpp](https://github.com/ggml-org/llama.cpp) (Llama, Mistral, Qwen, Gemma, Phi, etc.) by hooking the graph eval callback—no model source changes required.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
# Model metadata smoke test
./build/llm-trace --model path/to/model.gguf

# Live capture to stdout (non-invasive cb_eval hook)
./build/llm-trace --model path/to/model.gguf --verbose-trace --prompt "hello"

# Interactive TUI during live inference
./build/llm-trace --model path/to/model.gguf --prompt "hello" --tui

# Record a session trace
./build/llm-trace --model path/to/model.gguf --prompt "hello" --record session.trace --tui

# Replay without a model
./build/generate-demo-trace demo.trace
./build/llm-trace --replay demo.trace --tui
```

## TUI Keys

| Key | Action |
|-----|--------|
| `Tab` | Cycle panel focus |
| `Q` | Quit |
| `j` / `k` | Navigate lists |
| `Space` | Set capture target (topology) |
| `h` `j` `k` `l` | Pan attention matrix |
| `+` / `-` | Attention contrast |
| `[` / `]` | Change attention head |

## Requirements

- CMake 3.20+, C++17 compiler
- Live capture: a GGUF model (e.g. TinyLlama, Qwen2.5-0.5B)
- Replay mode: no model required
