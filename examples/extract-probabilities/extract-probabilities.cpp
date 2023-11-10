#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>

static int top_k = 100;

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct results_extraction {
    std::vector<llama_token> tokens;
    double                   ppl_value;
    std::vector<float>       logits;
    std::vector<float>       probs;
};

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const struct results_extraction & results
) {
    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: main\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, results.tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Perplexity Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_vector_float_yaml(logfile, "logits", results.logits);
    fprintf(logfile, "ppl_value: %f\n", results.ppl_value);
    dump_vector_float_yaml(logfile, "probs", results.probs);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

// Note: the probability values of softmax() and log_softmax() can be different in `perplexity.cpp`,
// Because the sum of the exponential logits is casted to float in `log_softmax()`, but not in `softmax()`.
//
// To address this issue, the attribute `cast_to_float` was added to `softmax()` and `log_softmax()`
static std::vector<float> softmax(const std::vector<float>& logits, bool cast_to_float = true) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        if(cast_to_float){
            probs[i] /= (float) sum_exp;
        } else {
            probs[i] /= sum_exp;
        }
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

/*
        Usage:

        const int first = n_ctx/2;
        process_logits(
            n_vocab, 
            const float * logits =        logits.data() + first*n_vocab =      logits.data() + n_ctx/2*n_vocab, 
            const int * tokens =       tokens.data() + start + first =      tokens.data() + i * n_ctx + n_ctx/2,
            int n_token =        n_ctx - 1 - first =       n_ctx - 1 - n_ctx/2,
            workers, 
            nll, 
            nll2, 
            float * logit_history =          logit_history.data() + start + first, 
            float * prob_history =          prob_history.data() + start + first,
            top_k_probs_history =         top_k_probs_history.data() + (start * top_k) = top_k_probs_history.data() + (i * n_ctx) * top_k
            );



    logits: pointer to the logits of the first token in the second half of the context window
*/
static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history, float * top_k_probs_history
) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token, top_k_probs_history] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static results_extraction extract_probabilities_v2(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const bool is_spm = llama_vocab_type(llama_get_model(ctx)) == LLAMA_VOCAB_TYPE_SPM;
    const bool add_bos = is_spm;

    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    const int n_ctx = llama_n_ctx(ctx);

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens to evaluate perplexity with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    logit_history.resize(tokens.size());
    prob_history.resize(tokens.size());

    if (params.ppl_stride <= 0) {
        fprintf(stderr, "%s: stride is %d but must be greater than zero!\n",__func__,params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int calc_chunk = n_ctx;

    fprintf(stderr, "%s: have %zu tokens. Calculation chunk = %d\n", __func__, tokens.size(), calc_chunk);

    if (int(tokens.size()) <= calc_chunk) {
        fprintf(stderr, "%s: there are only %zu tokens, this is not enough for a context size of %d and stride %d\n",__func__,
                tokens.size(), n_ctx, params.ppl_stride);
        return {tokens, -1, logit_history, prob_history};
    }

    const int n_chunk_max = (tokens.size() - calc_chunk + params.ppl_stride - 1)  / params.ppl_stride;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;

    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * params.ppl_stride;
        const int end   = start + calc_chunk;

        const int num_batches = (calc_chunk + n_batch - 1) / n_batch;
        //fprintf(stderr, "%s: evaluating %d...%d using %d batches\n", __func__, start, end, num_batches);

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            //fprintf(stderr, "    Batch %d: starts at %d, size is %d, n_past is %d\n",j,batch_start,batch_size,j * n_batch);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + batch_start, batch_size, j * n_batch, 0))) {
                //fprintf(stderr, "%s : failed to eval\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_token_bos(llama_get_model(ctx));
            }

            const auto batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);

            if (j == 0) {
                tokens[batch_start] = token_org;
            }
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%.2f minutes\n", total_seconds / 60.0);
        }

        //fprintf(stderr, "%s: using tokens %d...%d\n",__func__,params.n_ctx - params.ppl_stride + start, params.n_ctx + start);
        for (int j = n_ctx - params.ppl_stride - 1; j < n_ctx - 1; ++j) {

            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            bool cast_to_float = false; // do not cast, because it is not casted in `perplexity.cpp`
            const float prob = softmax(tok_logits, cast_to_float)[tokens[start + j + 1]];
            logit_history[start + j + 1] = tok_logits[tokens[start + j + 1]];
            prob_history[start + j + 1]  = prob;

            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            printf("%8d  %.4lf\n", i*params.ppl_stride, std::exp(nll / count));
        }
        fflush(stdout);
    }
    printf("\n");

    return {tokens, std::exp(nll / count), logit_history, prob_history};
}

static results_extraction extract_probabilities(llama_context * ctx, const gpt_params & params) {
    // TODO: change naming: ppl_stride
    if (params.ppl_stride > 0) {
        return extract_probabilities_v2(ctx, params);
    }

    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./calibration-raw -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `calibration-raw: 13.5106 [114/114]`
    // BOS tokens will be added for each chunk before eval

    const bool is_spm = llama_vocab_type(llama_get_model(ctx)) == LLAMA_VOCAB_TYPE_SPM;
    const bool add_bos = is_spm;
    const int n_ctx = llama_n_ctx(ctx);

    auto tim1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    auto tim2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens to extract probabilities with a context of %d\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return {std::move(tokens), 0., {}, {}};
    }

    // Collect logits and probabilities for each token in the input.
    std::vector<float> logit_history;
    logit_history.resize(tokens.size());

    std::vector<float> prob_history;
    prob_history.resize(tokens.size());

    // Initialize number of chunks, vocabulary size, and batch size
    const int n_chunk_max = tokens.size() / n_ctx; // Only full chunks are processed

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    // The task is to predict the next token given the previous ones.
    //
    // For each token, store `top_k` probabilities, including the correct one.
    // The correct probability will be the first one in the list.
    std::vector<float> top_k_probs_history;

    // Store probabilities only for the second half of the context window, because the first half is used for context.
    // The same is done in `perplexity.cpp`.
    const unsigned long top_k_probs_history_size = (n_ctx/2) * n_chunk * top_k; 
    top_k_probs_history.resize(top_k_probs_history_size);

    // top_k_probs_history_size 
    const float top_k_probs_history_size_in_megabytes = top_k_probs_history_size * sizeof(float) / 1024 / 1024;

    fprintf(stderr, "\ntop_k_probs_history_size_in_megabytes: %0.2fMB \n\n", top_k_probs_history_size_in_megabytes);

    // Initialize values for perplexity calculation
    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    fprintf(stderr, "%s: extracting probabilities over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    // Go over each chunk of the input. Chunk size is n_ctx.
    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx; // start index of the chunk
        const int end   = start + n_ctx;

        const int num_batches = (n_ctx + n_batch - 1) / n_batch;

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        // Go over each batch in the chunk. Batch size is n_batch.
        // Calculate logits for each token in the batch.
        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_token_bos(llama_get_model(ctx));
            }

            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + batch_start, batch_size, j * n_batch, 0))) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return {tokens, -1, logit_history, prob_history};
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            const auto * batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%.2f minutes\n", total_seconds / 60.0);
        }

        // FROM THE ORIGINAL CODE (perplexity.cpp):
        //
        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half of the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.
        const int first = n_ctx/2;

        int top_k_probs_history_start = (i * n_ctx) / 2 * top_k;


        process_logits(
            n_vocab, 
            logits.data() + first*n_vocab, 
            tokens.data() + start + first, 
            n_ctx - 1 - first,
            workers, 
            nll, 
            nll2, 
            logit_history.data() + start + first, 
            prob_history.data() + start + first,
            top_k_probs_history.data() + top_k_probs_history_start
            );
        count += n_ctx - first - 1;

        // perplexity is e^(average negative log-likelihood)
        if (params.ppl_output_type == 0) {
            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        } else {
            double av = nll/count;
            double av2 = nll2/count - av*av;
            if (av2 > 0) av2 = sqrt(av2/(count-1));
            printf("%8d  %.4lf  %4lf  %4lf\n", i*n_ctx, std::exp(nll / count), av, av2);
        }
        fflush(stdout);
    }
    printf("\n");

    nll2 /= count;
    nll /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        printf("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
    } else {
        printf("Unexpected negative standard deviation of log(prob)\n");
    }

    return {tokens, ppl, logit_history, prob_history};
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.n_batch = 512;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    params.logits_all = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.ppl_stride > 0) {
        fprintf(stderr, "Will perform strided perplexity calculation -> adjusting context size from %d to %d\n",
                params.n_ctx, params.n_ctx + params.ppl_stride/2);
        params.n_ctx += params.ppl_stride/2;
    }

    print_build_info();

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", get_system_info(params).c_str());
    }

    struct results_extraction results;
    results = extract_probabilities(ctx, params);

    llama_print_timings(ctx);
    write_logfile(ctx, params, model, results);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
