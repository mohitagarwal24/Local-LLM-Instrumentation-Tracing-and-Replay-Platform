# LLM Trace Platform

Non-invasive telemetry and tracing for local transformer models (GGUF via llama.cpp).

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
# Smoke test: print model metadata
./build/llm-trace --model path/to/model.gguf

# Live capture with TUI (coming soon)
./build/llm-trace --model path/to/model.gguf --prompt "hello"

# Replay a recorded trace (coming soon)
./build/llm-trace --replay trace.bin
```

Requires a GGUF model for live capture (e.g. TinyLlama, Qwen2.5-0.5B).
