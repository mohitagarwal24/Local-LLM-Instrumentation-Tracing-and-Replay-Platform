#include "engine/llama_runner.hpp"
#include "engine/model_load.hpp"
#include "tui/app.hpp"
#include "tracer/recorder.hpp"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --model <path.gguf> [--prompt <text>] [--tui] [--record <trace.bin>]\n"
                 "  %s --model <path.gguf> --verbose-trace [--prompt <text>]\n"
                 "  %s --replay <trace.bin> [--tui]\n",
                 argv0, argv0, argv0);
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

static int run_metadata_only(const std::string& model_path) {
    llama_backend_init();

    llama_model_params mparams = llm_trace_model_params();
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

int main(int argc, char** argv) {
    const std::string replay_path = arg_value(argc, argv, "--replay");
    const std::string model_path = arg_value(argc, argv, "--model");
    const std::string prompt = arg_value(argc, argv, "--prompt");
    const std::string record_path = arg_value(argc, argv, "--record");
    const bool verbose_trace = arg_flag(argc, argv, "--verbose-trace");
    const bool use_tui = arg_flag(argc, argv, "--tui");

    if (!replay_path.empty()) {
        trace::Replayer replayer(replay_path);
        if (!replayer.load()) {
            std::fprintf(stderr, "Failed to load trace: %s\n", replay_path.c_str());
            return 1;
        }
        if (use_tui) {
            tui::TraceApp app(nullptr, &replayer, false, {});
            return app.run();
        }
        std::printf("Replay loaded: %zu events, %zu anomalies, %zu attention snapshots\n",
                    replayer.events().size(), replayer.anomalies().size(),
                    replayer.attentions().size());
        return 0;
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (!verbose_trace && !use_tui && prompt.empty() && record_path.empty()) {
        return run_metadata_only(model_path);
    }

    engine::RunnerConfig rcfg;
    rcfg.model_path = model_path;
    rcfg.prompt = prompt.empty() ? "hello" : prompt;
    rcfg.verbose_trace = verbose_trace;

    engine::LlamaRunner runner(rcfg);
    if (!runner.init()) {
        return 1;
    }

    if (verbose_trace) {
        runner.start_worker();
        runner.request_decode();
        runner.wait();
        std::printf("Captured %zu trace events\n", runner.tracer().events().snapshot().size());
        return 0;
    }

    if (use_tui || !record_path.empty()) {
        tui::TraceApp app(&runner, nullptr, !record_path.empty(), record_path);
        return app.run();
    }

    runner.start_worker();
    runner.request_decode();
    runner.wait();
    return 0;
}
