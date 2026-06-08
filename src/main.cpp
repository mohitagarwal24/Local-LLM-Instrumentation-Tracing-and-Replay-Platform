#include <cstdio>
#include <cstring>
#include <string>

#include "llama.h"

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --model <path.gguf> [--prompt <text>] [--replay <trace.bin>]\n",
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

int main(int argc, char** argv) {
    const std::string model_path = arg_value(argc, argv, "--model");
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const int n_layer = llama_n_layer(model);
    const int n_head  = llama_n_head(model);
    const int n_embd  = llama_n_embd(model);
    const int n_vocab = llama_n_vocab(model);

    std::printf("Model loaded successfully\n");
    std::printf("  layers : %d\n", n_layer);
    std::printf("  heads  : %d\n", n_head);
    std::printf("  embd   : %d\n", n_embd);
    std::printf("  vocab  : %d\n", n_vocab);
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    std::printf("  desc   : %s\n", desc);

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
