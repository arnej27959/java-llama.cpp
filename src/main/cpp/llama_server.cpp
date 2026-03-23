#include "llama_server.h"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

// Note: server-task.h defines `using json = nlohmann::ordered_json;`
// so we use `json` as-is (ordered_json).

namespace {

void prepare_result_for_json(server_task_result_ptr& result) {
    if (result && !result->is_error()) {
        common_chat_parser_params parser_params;
        task_result_state         state(parser_params);
        result->update(state);
    }
}

} // namespace

LlamaServer::LlamaServer() = default;

LlamaServer::~LlamaServer() {
    shutdown();
}

void LlamaServer::load_model(const std::vector<std::string>& params) {
    // Build argc/argv from string vector
    std::vector<const char*> argv_ptrs;
    argv_ptrs.reserve(params.size());
    for (const auto& p : params) {
        argv_ptrs.push_back(p.c_str());
    }

    common_params cparams;
    const auto parsed = common_params_parse(static_cast<int>(argv_ptrs.size()), const_cast<char**>(argv_ptrs.data()),
                                            cparams, LLAMA_EXAMPLE_SERVER);
    if (!parsed) {
        throw std::runtime_error("failed to parse model parameters");
    }

    if (cparams.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);
        cparams.n_parallel = 4;
        cparams.kv_unified = true;
    }

    SRV_INF("loading model '%s'\n", cparams.model.path.c_str());

    common_init();

    ctx_server_ = new server_context();

    llama_numa_init(cparams.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", cparams.cpuparams.n_threads,
            cparams.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(cparams).c_str());
    LOG_INF("\n");
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server_->load_model(cparams)) {
        LOG_INF("%s: loading model failed\n", __func__);
        delete ctx_server_;
        ctx_server_ = nullptr;
        llama_backend_free();
        throw std::runtime_error("could not load model from given file path");
    }

    LOG_INF("%s: model loaded\n", __func__);

    const auto model_meta = ctx_server_->get_meta();

    // Initialize chat templates
    auto& chat_params = ctx_server_->get_chat_params();
    chat_params.tmpls = common_chat_templates_init(ctx_server_->get_model(), cparams.chat_template);
    chat_params.use_jinja = cparams.use_jinja;

    try {
        std::map<std::string, std::string> empty_kwargs;
        auto example_format = common_chat_format_example(model_meta.chat_params.tmpls.get(),
                                                         model_meta.chat_params.use_jinja, empty_kwargs);
        LOG_INF("%s: using chat_template %s: example_format: '%s'\n", __func__,
                common_chat_templates_source(model_meta.chat_params.tmpls.get()).c_str(), example_format.c_str());
    } catch (const std::exception& e) {
        SRV_WRN("%s: chat template not supported, falling back to chatml\n", __func__);
        chat_params.tmpls = common_chat_templates_init(ctx_server_->get_model(), "chatml");
    }
}

void LlamaServer::start(std::function<void()> pre_start) {
    if (!ctx_server_) {
        throw std::runtime_error("model not loaded");
    }
    loop_thread_ = std::thread([this, pre_start = std::move(pre_start)]() {
        if (pre_start) {
            pre_start();
        }
        ctx_server_->start_loop();
    });
}

void LlamaServer::shutdown() {
    if (!ctx_server_) {
        return;
    }
    auto model_name = ctx_server_->get_meta().model_name;
    LOG_INF("%s: deleting llama model %s, terminating server...\n", __func__, model_name.c_str());
    ctx_server_->get_queue_tasks().terminate();
    common_log_pause(common_log_main());
    ctx_server_->terminate();
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    delete ctx_server_;
    ctx_server_ = nullptr;
    LOG_INF("%s: deleted server for llama model '%s'\n", __func__, model_name.c_str());
}

int LlamaServer::request_completion(const std::string& json_params) {
    json data = json::parse(json_params);

    server_task_type type = SERVER_TASK_TYPE_COMPLETION;
    if (data.contains("input_prefix") || data.contains("input_suffix")) {
        type = SERVER_TASK_TYPE_INFILL;
    }

    auto                     completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;

    const auto&                prompt = data.at("prompt");
    std::vector<server_tokens> tokenized_prompts =
        tokenize_input_prompts(ctx_server_->get_vocab(), ctx_server_->get_mctx(), prompt, true, true);

    tasks.reserve(tokenized_prompts.size());
    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(type);

        task.id = ctx_server_->get_queue_tasks().get_new_id();
        task.index = i;
        task.tokens = std::move(tokenized_prompts[i]);
        task.params = server_task::params_from_json_cmpl(ctx_server_->get_vocab(), ctx_server_->get_params_base(),
                                                         ctx_server_->get_n_ctx_slot(), data);
        task.id_slot = json_value(data, "id_slot", -1);
        task.params.oaicompat_cmpl_id = completion_id;
        tasks.push_back(std::move(task));
    }

    for (const auto& task : tasks) {
        ctx_server_->get_queue_results().add_waiting_task_id(task.id);
    }
    const auto task_ids = server_task::get_list_id(tasks);
    ctx_server_->get_queue_tasks().post(std::move(tasks));

    if (task_ids.size() != 1) {
        throw std::runtime_error("multitasking currently not supported");
    }

    return *task_ids.begin();
}

CompletionResult LlamaServer::receive_completion(int task_id) {
    server_task_result_ptr result = ctx_server_->get_queue_results().recv(task_id);

    prepare_result_for_json(result);

    if (result->is_error()) {
        std::string msg = result->to_json()["message"].get<std::string>();
        ctx_server_->get_queue_results().remove_waiting_task_id(task_id);
        throw std::runtime_error(msg);
    }

    const auto out_res = result->to_json();

    CompletionResult cr;
    cr.text = out_res["content"].get<std::string>();
    cr.stop = result->is_stop();

    if (out_res.contains("completion_probabilities")) {
        for (const auto& entry : out_res["completion_probabilities"]) {
            for (const auto& tp : entry["probs"]) {
                cr.probabilities[tp["tok_str"].get<std::string>()] = tp["prob"].get<float>();
            }
        }
    }

    if (result->is_stop()) {
        ctx_server_->get_queue_results().remove_waiting_task_id(task_id);
    }

    return cr;
}

void LlamaServer::cancel_completion(int task_id) {
    ctx_server_->get_queue_results().remove_waiting_task_id(task_id);
}

void LlamaServer::release_task(int task_id) {
    ctx_server_->get_queue_results().remove_waiting_task_id(task_id);
}

std::vector<float> LlamaServer::embed(const std::string& prompt) {
    if (!ctx_server_->get_params_base().embedding) {
        throw std::runtime_error(
            "model was not loaded with embedding support (see ModelParameters#setEmbedding(boolean))");
    }

    SRV_INF("Calling embedding '%s'\n", prompt.c_str());

    const auto tokens_vec = tokenize_mixed(ctx_server_->get_vocab(), prompt, true, true);

    server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);
    task.id = ctx_server_->get_queue_tasks().get_new_id();
    task.index = 0;
    task.tokens = server_tokens(tokens_vec, false);

    std::vector<server_task> tasks;
    tasks.push_back(std::move(task));

    for (const auto& t : tasks) {
        ctx_server_->get_queue_results().add_waiting_task_id(t.id);
    }
    std::unordered_set<int> task_ids = server_task::get_list_id(tasks);
    ctx_server_->get_queue_tasks().post(std::move(tasks));

    const auto id_task = *task_ids.begin();

    server_task_result_ptr result = ctx_server_->get_queue_results().recv(id_task);
    prepare_result_for_json(result);

    if (result->is_error()) {
        std::string msg = result->to_json()["message"].get<std::string>();
        ctx_server_->get_queue_results().remove_waiting_task_id(id_task);
        throw std::runtime_error(msg);
    }

    if (result->is_stop()) {
        ctx_server_->get_queue_results().remove_waiting_task_id(id_task);
    }

    const auto                      out_res = result->to_json();
    std::vector<std::vector<float>> embedding = out_res["embedding"].get<std::vector<std::vector<float>>>();

    if (embedding.empty() || embedding[0].empty()) {
        throw std::runtime_error("embedding array is empty");
    }

    return std::move(embedding[0]);
}

std::map<std::string, float> LlamaServer::rerank(const std::string&              query,
                                                 const std::vector<std::string>& documents) {

    server_context_meta meta = ctx_server_->get_meta();
    if (!ctx_server_->get_params_base().embedding || meta.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        throw std::runtime_error("This server does not support reranking. Start it with `--reranking`");
    }

    // get a response reader (manages task IDs and result queues)
    server_response_reader rd = ctx_server_->get_response_reader();

    std::vector<server_task> tasks;
    tasks.reserve(documents.size());
    for (size_t i = 0; i < documents.size(); i++) {
        server_task task(SERVER_TASK_TYPE_RERANK);
        task.id     = rd.get_new_id();
        task.index  = i;
        task.tokens = format_prompt_rerank(
            ctx_server_->get_model(),
            ctx_server_->get_vocab(),
            ctx_server_->get_mctx(),
            query,
            documents[i]
        );
        tasks.push_back(std::move(task));
    }
    rd.post_tasks(std::move(tasks));

    // wait for all results (no HTTP connection to check, so never stop early)
    auto all_results = rd.wait_for_all([] { return false; });

    if (all_results.error) {
        auto msg = all_results.error->to_json()["message"].get<std::string>();
        throw std::runtime_error(msg);
    }

    std::map<std::string, float> scores;
    for (auto& result : all_results.results) {
        const auto out_res = result->to_json();
        int        index = out_res["index"].get<int>();
        float      score = out_res["score"].get<float>();
        scores[documents[index]] = score;
    }

    return scores;
}

std::vector<int> LlamaServer::encode(const std::string& text) {
    llama_tokens tokens = tokenize_mixed(ctx_server_->get_vocab(), text, false, true);
    return std::vector<int>(tokens.begin(), tokens.end());
}

std::string LlamaServer::decode(const std::vector<int>& tokens) {
    std::vector<llama_token> ltokens(tokens.begin(), tokens.end());
    return tokens_to_str(ctx_server_->get_llama_context(), ltokens);
}

std::string LlamaServer::apply_template(const std::string& json_params) {
    json                    data = json::parse(json_params);
    std::vector<raw_buffer> dummy_files;
    json template_data = oaicompat_chat_params_parse(data, ctx_server_->get_chat_params(), dummy_files);
    return template_data.at("prompt").get<std::string>();
}
