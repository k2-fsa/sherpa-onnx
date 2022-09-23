#include <vector>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <fstream>

#include "fbank_features.h"
#include "rnnt_beam_search.h"

#include "kaldi-native-fbank/csrc/online-feature.h"


int main(int argc, char* argv[]) {
    char* filename = argv[1];
    std::string search_method = argv[2];
    int num_active_paths = atoi(argv[3]);

    // General parameters
    int numberOfThreads = 16;

    // Initialize fbanks
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0;
    opts.frame_opts.samp_freq = 16000;
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.mel_opts.num_bins = 80;
    opts.frame_opts.window_type = "povey";
    opts.frame_opts.snip_edges = false;
    knf::OnlineFbank fbank(opts);

    // set session opts
    // https://onnxruntime.ai/docs/performance/tune-performance.html
    session_options.SetIntraOpNumThreads(numberOfThreads);
    session_options.SetInterOpNumThreads(numberOfThreads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(4);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    api.CreateTensorRTProviderOptions(&tensorrt_options);
    std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
    api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get());

    // Define model
    auto model = get_model(
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/encoder_simp.onnx",
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/decoder_simp.onnx",
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/joiner_simp.onnx",
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/joiner_encoder_proj_simp.onnx",
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/joiner_decoder_proj_simp.onnx",
        "/mnt/local4/sr/k2_sherpa/models/exp_en2/enUS_tokens.txt"
    );
    
    std::vector<std::string> filename_list {
        "/mnt/local4/sr/k2_sherpa/test_wavs/cnn_15sec.wav",
        //"/mnt/local4/sr/k2_sherpa/test_wavs/1089-134686-0001.wav"
    };

    for (auto filename : filename_list){
        std::cout << filename << std::endl;
        auto samples = readWav(filename, true);
        int numSamples = samples.NumCols();

        auto features = ComputeFeatures(fbank, opts, samples);

        auto tic = std::chrono::high_resolution_clock::now();

        // # === Encoder Out === #
        int num_frames = features.size() / opts.mel_opts.num_bins;
        auto encoder_out = model.encoder_forward(features,
                                std::vector<int64_t> {num_frames},
                                std::vector<int64_t> {1, num_frames, 80},
                                std::vector<int64_t> {1},
                                memory_info);

        // # === Search === #
        std::vector<std::vector<int32_t>> hyps;
        if (search_method == "greedy")
            hyps = GreedySearch(&model, &encoder_out);
        else{
            std::cout << "wrong search method!" << std::endl;
            exit(0);
        }
        auto results = hyps2result(model.tokens_map, hyps);

        // # === Print Elapsed Time === #
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tic);
        std::cout << "Elapsed: " << float(elapsed.count()) / 1000 << " seconds" << std::endl;
        std::cout << "rtf: " << float(elapsed.count()) / 1000 / (numSamples / 16000)  << std::endl;

        print_hyps(hyps);
        std::cout << results[0] << std::endl;
    }

    return 0;
}
