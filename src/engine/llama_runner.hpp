#pragma once

#include "tracer/tracer.hpp"
#include "tracer/topology.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <thread>

struct llama_context;
struct llama_model;

namespace engine {

struct RunnerConfig {
    std::string model_path;
    std::string prompt = "hello";
    int n_ctx = 512;
    int n_threads = 4;
    bool verbose_trace = false;
};

class LlamaRunner {
public:
    explicit LlamaRunner(RunnerConfig config);
    ~LlamaRunner();

    LlamaRunner(const LlamaRunner&) = delete;
    LlamaRunner& operator=(const LlamaRunner&) = delete;

    bool init();
    void start_async();
    void wait();
    void request_stop();

    trace::Tracer& tracer() { return tracer_; }
    const trace::Tracer& tracer() const { return tracer_; }
    trace::Topology& topology() { return topology_; }
    const trace::Topology& topology() const { return topology_; }

    bool running() const { return running_.load(); }
    bool finished() const { return finished_.load(); }
    std::string model_name() const { return model_name_; }

private:
    void run_once();

    RunnerConfig config_;
    trace::Tracer tracer_;
    trace::Topology topology_;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    std::string model_name_;

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> finished_{false};
    std::atomic<bool> stop_{false};
};

}  // namespace engine
