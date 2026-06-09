#include "engine/llama_runner.hpp"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --model <path.gguf> [--prompt <text>] [--verbose-trace]\n",
                 argv0);
}

static std::string arg_value(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return argv[i + 1];
        }
    }
    return {};
}

static bool arg_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    const std::string model_path = arg_value(argc, argv, "--model");
    const std::string prompt = arg_value(argc, argv, "--prompt");
    const bool verbose_trace = arg_flag(argc, argv, "--verbose-trace");

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (!verbose_trace && prompt.empty()) {
        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        llama_model* model = llama_load_model_from_file(model_path.c_str(), mparams);
        if (!model) {
            std::fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
            llama_backend_free();
            return 1;
        }
        char desc[256] = {};
        llama_model_desc(model, desc, sizeof(desc));
        std::printf("Model loaded successfully\n");
        std::printf("  layers : %d\n", llama_n_layer(model));
        std::printf("  heads  : %d\n", llama_n_head(model));
        std::printf("  embd   : %d\n", llama_n_embd(model));
        std::printf("  vocab  : %d\n", llama_n_vocab(model));
        std::printf("  desc   : %s\n", desc);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }

    engine::RunnerConfig rcfg;
    rcfg.model_path = model_path;
    rcfg.prompt = prompt.empty() ? "hello" : prompt;
    rcfg.verbose_trace = true;

    engine::LlamaRunner runner(rcfg);
    if (!runner.init()) {
        return 1;
    }

    runner.start_async();
    runner.wait();
    std::printf("Captured %zu trace events\n", runner.tracer().events().snapshot().size());
    return 0;
}
