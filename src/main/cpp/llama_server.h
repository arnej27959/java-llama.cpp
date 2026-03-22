#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct server_context;

struct CompletionResult {
    std::string text;
    std::map<std::string, float> probabilities;
    bool stop;
};

class LlamaServer {
public:
    LlamaServer();
    ~LlamaServer();

    LlamaServer(const LlamaServer &) = delete;
    LlamaServer &operator=(const LlamaServer &) = delete;

    // Load model from CLI-style string params. Does NOT start the processing loop.
    void load_model(const std::vector<std::string> &params);

    // Start background processing loop in an owned thread.
    // pre_start is called on the new thread before entering the loop (e.g. JVM attach).
    void start(std::function<void()> pre_start = nullptr);

    // Terminate queues, join background thread, delete context. Safe to call multiple times.
    void shutdown();

    int request_completion(const std::string &json_params);
    CompletionResult receive_completion(int task_id);
    void cancel_completion(int task_id);
    void release_task(int task_id);

    std::vector<float> embed(const std::string &prompt);
    std::map<std::string, float> rerank(const std::string &query, const std::vector<std::string> &documents);

    std::vector<int> encode(const std::string &text);
    std::string decode(const std::vector<int> &tokens);
    std::string apply_template(const std::string &json_params);

private:
    server_context *ctx_server_ = nullptr;
    std::thread loop_thread_;
};
