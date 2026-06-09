#include "engine/llama_runner.hpp"

#include "llama.h"

#include <cstdio>
#include <vector>

namespace engine {

LlamaRunner::LlamaRunner(RunnerConfig config) : config_(std::move(config)), tracer_() {
    tracer_.set_verbose_stdout(config_.verbose_trace);
}

LlamaRunner::~LlamaRunner() {
    request_stop();
    wait();
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_free_model(model_);
        model_ = nullptr;
    }
}

bool LlamaRunner::init() {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    model_ = llama_load_model_from_file(config_.model_path.c_str(), mparams);
    if (!model_) {
        std::fprintf(stderr, "Failed to load model: %s\n", config_.model_path.c_str());
        return false;
    }

    const auto slash = config_.model_path.find_last_of("/\\");
    model_name_ = slash == std::string::npos ? config_.model_path
                                             : config_.model_path.substr(slash + 1);
    topology_.build_from_model(model_, model_name_);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(config_.n_ctx);
    cparams.n_threads = config_.n_threads;
    cparams.n_threads_batch = config_.n_threads;
    cparams.flash_attn = false;
    cparams.cb_eval = trace::Tracer::eval_callback;
    cparams.cb_eval_user_data = &tracer_;

    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) {
        std::fprintf(stderr, "Failed to create llama context\n");
        return false;
    }

    return true;
}

void LlamaRunner::start_async() {
    if (running_.load()) {
        return;
    }
    stop_.store(false);
    finished_.store(false);
    running_.store(true);
    worker_ = std::thread([this] { run_once(); });
}

void LlamaRunner::wait() {
    if (worker_.joinable()) {
        worker_.join();
    }
}

void LlamaRunner::request_stop() {
    stop_.store(true);
}

void LlamaRunner::run_once() {
    if (!ctx_) {
        running_.store(false);
        finished_.store(true);
        return;
    }

    tracer_.reset();

    const int n_vocab = llama_n_vocab(model_);
    std::vector<llama_token> tokens(config_.prompt.size() + 8);
    const int n_tokens = llama_tokenize(model_, config_.prompt.c_str(),
                                        static_cast<int>(config_.prompt.size()),
                                        tokens.data(), static_cast<int>(tokens.size()),
                                        true, false);
    if (n_tokens < 0) {
        std::fprintf(stderr, "Tokenization failed\n");
        running_.store(false);
        finished_.store(true);
        return;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx_, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
    }

    (void)n_vocab;
    running_.store(false);
    finished_.store(true);
}

}  // namespace engine
