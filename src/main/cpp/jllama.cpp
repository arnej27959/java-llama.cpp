#include "jllama.h"
#include "llama_server.h"

#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>

// JNI class/method/field caches and helper functions.
// Initialized in JNI_OnLoad, released in JNI_OnUnload.

namespace {
JavaVM *g_vm = nullptr;

// classes
jclass c_llama_model = nullptr;
jclass c_llama_iterator = nullptr;
jclass c_standard_charsets = nullptr;
jclass c_output = nullptr;
jclass c_string = nullptr;
jclass c_hash_map = nullptr;
jclass c_map = nullptr;
jclass c_set = nullptr;
jclass c_entry = nullptr;
jclass c_iterator = nullptr;
jclass c_integer = nullptr;
jclass c_float = nullptr;
jclass c_biconsumer = nullptr;
jclass c_llama_error = nullptr;
jclass c_log_level = nullptr;
jclass c_log_format = nullptr;
jclass c_error_oom = nullptr;

// constructors
jmethodID cc_output = nullptr;
jmethodID cc_hash_map = nullptr;
jmethodID cc_integer = nullptr;
jmethodID cc_float = nullptr;

// methods
jmethodID m_get_bytes = nullptr;
jmethodID m_entry_set = nullptr;
jmethodID m_set_iterator = nullptr;
jmethodID m_iterator_has_next = nullptr;
jmethodID m_iterator_next = nullptr;
jmethodID m_entry_key = nullptr;
jmethodID m_entry_value = nullptr;
jmethodID m_map_put = nullptr;
jmethodID m_int_value = nullptr;
jmethodID m_float_value = nullptr;
jmethodID m_biconsumer_accept = nullptr;

// fields
jfieldID f_model_pointer = nullptr;
jfieldID f_task_id = nullptr;
jfieldID f_utf_8 = nullptr;
jfieldID f_iter_has_next = nullptr;
jfieldID f_log_level_debug = nullptr;
jfieldID f_log_level_info = nullptr;
jfieldID f_log_level_warn = nullptr;
jfieldID f_log_level_error = nullptr;
jfieldID f_log_format_json = nullptr;
jfieldID f_log_format_text = nullptr;

// objects
jobject o_utf_8 = nullptr;
jobject o_log_level_debug = nullptr;
jobject o_log_level_info = nullptr;
jobject o_log_level_warn = nullptr;
jobject o_log_level_error = nullptr;
jobject o_log_format_json = nullptr;
jobject o_log_format_text = nullptr;
jobject o_log_callback = nullptr;

std::string parse_jstring(JNIEnv *env, jstring java_string) {
    auto *const string_bytes = (jbyteArray)env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);
    auto length = (size_t)env->GetArrayLength(string_bytes);
    jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);
    std::string string = std::string((char *)byte_elements, length);
    env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
    env->DeleteLocalRef(string_bytes);
    return string;
}

char **parse_string_array(JNIEnv *env, const jobjectArray string_array, const jsize length) {
    auto *const result = static_cast<char **>(malloc(length * sizeof(char *)));
    if (result == nullptr) {
        return nullptr;
    }
    for (jsize i = 0; i < length; i++) {
        auto *const javaString = static_cast<jstring>(env->GetObjectArrayElement(string_array, i));
        const char *cString = env->GetStringUTFChars(javaString, nullptr);
        result[i] = strdup(cString);
        env->ReleaseStringUTFChars(javaString, cString);
    }
    return result;
}

void free_string_array(char **array, jsize length) {
    if (array != nullptr) {
        for (jsize i = 0; i < length; i++) {
            free(array[i]);
        }
        free(array);
    }
}

jbyteArray parse_jbytes(JNIEnv *env, const std::string &string) {
    jsize length = string.size(); // NOLINT(*-narrowing-conversions)
    jbyteArray bytes = env->NewByteArray(length);
    env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte *>(string.c_str()));
    return bytes;
}

jobject log_level_to_jobject(ggml_log_level level) {
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        return o_log_level_error;
    case GGML_LOG_LEVEL_WARN:
        return o_log_level_warn;
    default:
    case GGML_LOG_LEVEL_INFO:
        return o_log_level_info;
    case GGML_LOG_LEVEL_DEBUG:
        return o_log_level_debug;
    }
}

JNIEnv *get_jni_env() {
    JNIEnv *env = nullptr;
    if (g_vm == nullptr || g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        throw std::runtime_error("Thread is not attached to the JVM");
    }
    return env;
}

bool log_json;
std::function<void(ggml_log_level, const char *, void *)> log_callback;

void log_callback_trampoline(ggml_log_level level, const char *text, void *user_data) {
    if (log_callback != nullptr) {
        log_callback(level, text, user_data);
    }
}

jint throwJava(JNIEnv *env, const char *message) {
    if (env && c_llama_error && message) {
        return env->ThrowNew(c_llama_error, message);
    }
    return JNI_ERR;
}

LlamaServer *getLlamaServerOrThrow(JNIEnv *env, jobject obj) {
    if (!f_model_pointer) {
        throwJava(env, "missing initialization");
        return nullptr;
    }
    jlong handle = env->GetLongField(obj, f_model_pointer);
    if (handle == 0) {
        throwJava(env, "no model loaded");
        return nullptr;
    }
    return reinterpret_cast<LlamaServer *>(handle); // NOLINT(*-no-int-to-ptr)
}

} // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    fprintf(stderr, "LlamaModel JNI_OnLoad()\n");
    g_vm = vm;
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_1)) {
        goto error;
    }

    // set up error handling
    c_llama_error = env->FindClass("de/kherud/llama/LlamaException");
    if (!c_llama_error) {
        goto error;
    }
    c_llama_error = (jclass)env->NewGlobalRef(c_llama_error);
    if (!c_llama_error) {
        goto error;
    }

    // find classes
    c_llama_model = env->FindClass("de/kherud/llama/LlamaModel");
    c_llama_iterator = env->FindClass("de/kherud/llama/LlamaIterator");
    c_standard_charsets = env->FindClass("java/nio/charset/StandardCharsets");
    c_output = env->FindClass("de/kherud/llama/LlamaOutput");
    c_string = env->FindClass("java/lang/String");
    c_hash_map = env->FindClass("java/util/HashMap");
    c_map = env->FindClass("java/util/Map");
    c_set = env->FindClass("java/util/Set");
    c_entry = env->FindClass("java/util/Map$Entry");
    c_iterator = env->FindClass("java/util/Iterator");
    c_integer = env->FindClass("java/lang/Integer");
    c_float = env->FindClass("java/lang/Float");
    c_biconsumer = env->FindClass("java/util/function/BiConsumer");
    c_log_level = env->FindClass("de/kherud/llama/LogLevel");
    c_log_format = env->FindClass("de/kherud/llama/args/LogFormat");
    c_error_oom = env->FindClass("java/lang/OutOfMemoryError");

    if (!(c_llama_model && c_llama_iterator && c_standard_charsets && c_output && c_string && c_hash_map && c_map &&
          c_set && c_entry && c_iterator && c_integer && c_float && c_biconsumer && c_llama_error && c_log_level &&
          c_log_format && c_error_oom)) {
        goto error;
    }

    // create references
    c_llama_model = (jclass)env->NewGlobalRef(c_llama_model);
    c_llama_iterator = (jclass)env->NewGlobalRef(c_llama_iterator);
    c_output = (jclass)env->NewGlobalRef(c_output);
    c_string = (jclass)env->NewGlobalRef(c_string);
    c_hash_map = (jclass)env->NewGlobalRef(c_hash_map);
    c_map = (jclass)env->NewGlobalRef(c_map);
    c_set = (jclass)env->NewGlobalRef(c_set);
    c_entry = (jclass)env->NewGlobalRef(c_entry);
    c_iterator = (jclass)env->NewGlobalRef(c_iterator);
    c_integer = (jclass)env->NewGlobalRef(c_integer);
    c_float = (jclass)env->NewGlobalRef(c_float);
    c_biconsumer = (jclass)env->NewGlobalRef(c_biconsumer);
    c_log_level = (jclass)env->NewGlobalRef(c_log_level);
    c_log_format = (jclass)env->NewGlobalRef(c_log_format);
    c_error_oom = (jclass)env->NewGlobalRef(c_error_oom);

    // find constructors
    cc_output = env->GetMethodID(c_output, "<init>", "([BLjava/util/Map;Z)V");
    cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
    cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
    cc_float = env->GetMethodID(c_float, "<init>", "(F)V");

    if (!(cc_output && cc_hash_map && cc_integer && cc_float)) {
        goto error;
    }

    // find methods
    m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
    m_entry_set = env->GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
    m_set_iterator = env->GetMethodID(c_set, "iterator", "()Ljava/util/Iterator;");
    m_iterator_has_next = env->GetMethodID(c_iterator, "hasNext", "()Z");
    m_iterator_next = env->GetMethodID(c_iterator, "next", "()Ljava/lang/Object;");
    m_entry_key = env->GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
    m_entry_value = env->GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
    m_map_put = env->GetMethodID(c_map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    m_int_value = env->GetMethodID(c_integer, "intValue", "()I");
    m_float_value = env->GetMethodID(c_float, "floatValue", "()F");
    m_biconsumer_accept = env->GetMethodID(c_biconsumer, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");

    if (!(m_get_bytes && m_entry_set && m_set_iterator && m_iterator_has_next && m_iterator_next && m_entry_key &&
          m_entry_value && m_map_put && m_int_value && m_float_value && m_biconsumer_accept)) {
        goto error;
    }

    // find fields
    f_model_pointer = env->GetFieldID(c_llama_model, "ctx", "J");
    f_task_id = env->GetFieldID(c_llama_iterator, "taskId", "I");
    f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
    f_iter_has_next = env->GetFieldID(c_llama_iterator, "hasNext", "Z");
    f_log_level_debug = env->GetStaticFieldID(c_log_level, "DEBUG", "Lde/kherud/llama/LogLevel;");
    f_log_level_info = env->GetStaticFieldID(c_log_level, "INFO", "Lde/kherud/llama/LogLevel;");
    f_log_level_warn = env->GetStaticFieldID(c_log_level, "WARN", "Lde/kherud/llama/LogLevel;");
    f_log_level_error = env->GetStaticFieldID(c_log_level, "ERROR", "Lde/kherud/llama/LogLevel;");
    f_log_format_json = env->GetStaticFieldID(c_log_format, "JSON", "Lde/kherud/llama/args/LogFormat;");
    f_log_format_text = env->GetStaticFieldID(c_log_format, "TEXT", "Lde/kherud/llama/args/LogFormat;");

    if (!(f_model_pointer && f_task_id && f_utf_8 && f_iter_has_next && f_log_level_debug && f_log_level_info &&
          f_log_level_warn && f_log_level_error && f_log_format_json && f_log_format_text)) {
        goto error;
    }

    o_utf_8 = env->NewStringUTF("UTF-8");
    o_log_level_debug = env->GetStaticObjectField(c_log_level, f_log_level_debug);
    o_log_level_info = env->GetStaticObjectField(c_log_level, f_log_level_info);
    o_log_level_warn = env->GetStaticObjectField(c_log_level, f_log_level_warn);
    o_log_level_error = env->GetStaticObjectField(c_log_level, f_log_level_error);
    o_log_format_json = env->GetStaticObjectField(c_log_format, f_log_format_json);
    o_log_format_text = env->GetStaticObjectField(c_log_format, f_log_format_text);

    if (!(o_utf_8 && o_log_level_debug && o_log_level_info && o_log_level_warn && o_log_level_error &&
          o_log_format_json && o_log_format_text)) {
        goto error;
    }

    o_utf_8 = env->NewGlobalRef(o_utf_8);
    o_log_level_debug = env->NewGlobalRef(o_log_level_debug);
    o_log_level_info = env->NewGlobalRef(o_log_level_info);
    o_log_level_warn = env->NewGlobalRef(o_log_level_warn);
    o_log_level_error = env->NewGlobalRef(o_log_level_error);
    o_log_format_json = env->NewGlobalRef(o_log_format_json);
    o_log_format_text = env->NewGlobalRef(o_log_format_text);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        goto error;
    }

    llama_backend_init();

    goto success;

error:
    return JNI_ERR;

success:
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    fprintf(stderr, "LlamaModel JNI_OnUnLoad()\n");
    JNIEnv *env = nullptr;

    if (JNI_OK != vm->GetEnv((void **)&env, JNI_VERSION_1_6)) {
        return;
    }

    env->DeleteGlobalRef(c_llama_model);
    env->DeleteGlobalRef(c_llama_iterator);
    env->DeleteGlobalRef(c_output);
    env->DeleteGlobalRef(c_string);
    env->DeleteGlobalRef(c_hash_map);
    env->DeleteGlobalRef(c_map);
    env->DeleteGlobalRef(c_set);
    env->DeleteGlobalRef(c_entry);
    env->DeleteGlobalRef(c_iterator);
    env->DeleteGlobalRef(c_integer);
    env->DeleteGlobalRef(c_float);
    env->DeleteGlobalRef(c_biconsumer);
    env->DeleteGlobalRef(c_llama_error);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_log_level);
    env->DeleteGlobalRef(c_error_oom);

    env->DeleteGlobalRef(o_utf_8);
    env->DeleteGlobalRef(o_log_level_debug);
    env->DeleteGlobalRef(o_log_level_info);
    env->DeleteGlobalRef(o_log_level_warn);
    env->DeleteGlobalRef(o_log_level_error);
    env->DeleteGlobalRef(o_log_format_json);
    env->DeleteGlobalRef(o_log_format_text);

    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    llama_backend_free();
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_loadModel(JNIEnv *env, jobject obj, jobjectArray jparams) {
    fprintf(stderr, "LlamaModel_loadModel()\n");

    const jsize argc = env->GetArrayLength(jparams);
    char **argv = parse_string_array(env, jparams, argc);
    if (argv == nullptr) {
        return;
    }

    std::vector<std::string> params_vec;
    params_vec.reserve(argc);
    for (jsize i = 0; i < argc; i++) {
        params_vec.emplace_back(argv[i]);
    }
    free_string_array(argv, argc);

    try {
        auto *server = new LlamaServer();
        server->load_model(params_vec);
        server->start([&]() {
            JNIEnv *tenv;
            jint res = g_vm->GetEnv((void **)&tenv, JNI_VERSION_1_6);
            if (res == JNI_EDETACHED) {
                res = g_vm->AttachCurrentThread((void **)&tenv, nullptr);
                if (res != JNI_OK) {
                    throw std::runtime_error("Failed to attach thread to JVM");
                }
            }
        });
        env->SetLongField(obj, f_model_pointer, reinterpret_cast<jlong>(server));
    } catch (const std::exception &e) {
        throwJava(env, e.what());
    }
}

JNIEXPORT jint JNICALL Java_de_kherud_llama_LlamaModel_requestCompletion(JNIEnv *env, jobject obj, jstring jparams) {
    fprintf(stderr, "LlamaModel_requestCompletion()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return 0;

    try {
        std::string params = parse_jstring(env, jparams);
        int task_id = server->request_completion(params);
        fprintf(stderr, "LlamaModel_requestCompletion -> %d\n", task_id);
        return task_id;
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_releaseTask(JNIEnv *env, jobject obj, jint id_task) {
    fprintf(stderr, "LlamaModel_releaseTask()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return;
    server->release_task(id_task);
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_receiveCompletion(JNIEnv *env, jobject obj, jint id_task) {
    fprintf(stderr, "LlamaModel_receiveCompletion()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        CompletionResult cr = server->receive_completion(id_task);

        jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
        for (const auto &[tok_str, prob] : cr.probabilities) {
            jstring jtok_str = env->NewStringUTF(tok_str.c_str());
            jobject jprob = env->NewObject(c_float, cc_float, prob);
            env->CallObjectMethod(o_probabilities, m_map_put, jtok_str, jprob);
            env->DeleteLocalRef(jtok_str);
            env->DeleteLocalRef(jprob);
        }

        jbyteArray jbytes = parse_jbytes(env, cr.text);
        return env->NewObject(c_output, cc_output, jbytes, o_probabilities, cr.stop);
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jfloatArray JNICALL Java_de_kherud_llama_LlamaModel_embed(JNIEnv *env, jobject obj, jstring jprompt) {
    fprintf(stderr, "LlamaModel_embed()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        std::string prompt = parse_jstring(env, jprompt);
        std::vector<float> embedding = server->embed(prompt);

        jfloatArray j_embedding = env->NewFloatArray(static_cast<jsize>(embedding.size()));
        if (j_embedding == nullptr) {
            env->ThrowNew(c_error_oom, "could not allocate embedding");
            return nullptr;
        }
        env->SetFloatArrayRegion(j_embedding, 0, static_cast<jsize>(embedding.size()),
                                 reinterpret_cast<const jfloat *>(embedding.data()));
        return j_embedding;
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL Java_de_kherud_llama_LlamaModel_rerank(JNIEnv *env, jobject obj, jstring jprompt,
                                                                  jobjectArray documents) {
    fprintf(stderr, "LlamaModel_rerank()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        std::string query = parse_jstring(env, jprompt);

        const jsize amount_documents = env->GetArrayLength(documents);
        auto *document_array = parse_string_array(env, documents, amount_documents);
        std::vector<std::string> document_vector(document_array, document_array + amount_documents);
        free_string_array(document_array, amount_documents);

        std::map<std::string, float> scores = server->rerank(query, document_vector);

        jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
        if (o_probabilities == nullptr) {
            throwJava(env, "Failed to create HashMap object.");
            return nullptr;
        }

        for (const auto &[doc, score] : scores) {
            jstring jtok_str = env->NewStringUTF(doc.c_str());
            jobject jprob = env->NewObject(c_float, cc_float, score);
            env->CallObjectMethod(o_probabilities, m_map_put, jtok_str, jprob);
            env->DeleteLocalRef(jtok_str);
            env->DeleteLocalRef(jprob);
        }

        jbyteArray jbytes = parse_jbytes(env, query);
        return env->NewObject(c_output, cc_output, jbytes, o_probabilities, true);
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_de_kherud_llama_LlamaModel_applyTemplate(JNIEnv *env, jobject obj, jstring jparams) {
    fprintf(stderr, "LlamaModel_applyTemplate()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        std::string params = parse_jstring(env, jparams);
        std::string result = server->apply_template(params);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jintArray JNICALL Java_de_kherud_llama_LlamaModel_encode(JNIEnv *env, jobject obj, jstring jprompt) {
    fprintf(stderr, "LlamaModel_encode()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        std::string prompt = parse_jstring(env, jprompt);
        std::vector<int> tokens = server->encode(prompt);

        jsize token_size = static_cast<jsize>(tokens.size());
        jintArray java_tokens = env->NewIntArray(token_size);
        if (java_tokens == nullptr) {
            env->ThrowNew(c_error_oom, "could not allocate token memory");
            return nullptr;
        }
        env->SetIntArrayRegion(java_tokens, 0, token_size, reinterpret_cast<const jint *>(tokens.data()));
        return java_tokens;
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_decodeBytes(JNIEnv *env, jobject obj,
                                                                          jintArray java_tokens) {
    fprintf(stderr, "LlamaModel_decode()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return nullptr;

    try {
        jsize length = env->GetArrayLength(java_tokens);
        jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
        std::vector<int> tokens(elements, elements + length);
        env->ReleaseIntArrayElements(java_tokens, elements, 0);

        std::string text = server->decode(tokens);
        return parse_jbytes(env, text);
    } catch (const std::exception &e) {
        throwJava(env, e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_delete(JNIEnv *env, jobject obj) {
    fprintf(stderr, "LlamaModel_delete()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return;
    server->shutdown();
    delete server;
    env->SetLongField(obj, f_model_pointer, 0);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_cancelCompletion(JNIEnv *env, jobject obj, jint id_task) {
    fprintf(stderr, "LlamaModel_cancelCompletion()\n");
    auto *server = getLlamaServerOrThrow(env, obj);
    if (!server) return;
    server->cancel_completion(id_task);
}

JNIEXPORT void JNICALL Java_de_kherud_llama_LlamaModel_setLogger(JNIEnv *env, jclass clazz, jobject log_format,
                                                                  jobject jcallback) {
    fprintf(stderr, "LlamaModel_setLogger()\n");
    if (o_log_callback != nullptr) {
        env->DeleteGlobalRef(o_log_callback);
    }

    log_json = env->IsSameObject(log_format, o_log_format_json);

    if (jcallback == nullptr) {
        log_callback = nullptr;
        llama_log_set(nullptr, nullptr);
    } else {
        o_log_callback = env->NewGlobalRef(jcallback);
        log_callback = [](enum ggml_log_level level, const char *text, void *user_data) {
            JNIEnv *env = get_jni_env();
            jstring message = env->NewStringUTF(text);
            jobject log_level = log_level_to_jobject(level);
            env->CallVoidMethod(o_log_callback, m_biconsumer_accept, log_level, message);
            env->DeleteLocalRef(message);
        };
        if (!log_json) {
            llama_log_set(log_callback_trampoline, nullptr);
        }
    }
}

JNIEXPORT jbyteArray JNICALL Java_de_kherud_llama_LlamaModel_jsonSchemaToGrammarBytes(JNIEnv *env, jclass clazz,
                                                                                       jstring j_schema) {
    fprintf(stderr, "LlamaModel_jsonSchemaToGrammarBytes()\n");
    try {
        const std::string c_schema = parse_jstring(env, j_schema);
        nlohmann::ordered_json c_schema_json = nlohmann::ordered_json::parse(c_schema);
        const std::string c_grammar = json_schema_to_grammar(c_schema_json);
        return parse_jbytes(env, c_grammar);
    } catch (std::exception &ex) {
        const char *msg = ex.what();
        if (msg) {
            throwJava(env, msg);
        } else {
            throwJava(env, "jsonSchemaToGrammarBytes: Got C++ exception without any message");
        }
        return nullptr;
    }
}
