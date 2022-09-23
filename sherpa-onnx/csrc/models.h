#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>

#include "utils_onnx.h"


struct Model
{
    public:
        const char* encoder_path;
        const char* decoder_path;
        const char* joiner_path;
        const char* joiner_encoder_proj_path;
        const char* joiner_decoder_proj_path;
        const char* tokens_path;

        Ort::Session encoder = load_model(encoder_path);
        Ort::Session decoder = load_model(decoder_path);
        Ort::Session joiner = load_model(joiner_path);
        Ort::Session joiner_encoder_proj = load_model(joiner_encoder_proj_path);
        Ort::Session joiner_decoder_proj = load_model(joiner_decoder_proj_path);
        std::map<int, std::string> tokens_map = get_token_map(tokens_path);
        
        int32_t blank_id;
        int32_t unk_id;
        int32_t context_size;

        std::vector<Ort::Value> encoder_forward(std::vector<float> in_vector, 
                                            std::vector<int64_t> in_vector_length, 
                                            std::vector<int64_t> feature_dims, 
                                            std::vector<int64_t> feature_length_dims, 
                                            Ort::MemoryInfo &memory_info){
        std::vector<Ort::Value> encoder_inputTensors;
        encoder_inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, in_vector.data(), in_vector.size(), feature_dims.data(), feature_dims.size()));
        encoder_inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, in_vector_length.data(), in_vector_length.size(), feature_length_dims.data(), feature_length_dims.size()));
        
        std::vector<const char*> encoder_inputNames = {encoder.GetInputName(0, allocator), encoder.GetInputName(1, allocator)};
        std::vector<const char*> encoder_outputNames = {encoder.GetOutputName(0, allocator)};

        auto out = encoder.Run(Ort::RunOptions{nullptr}, 
                                encoder_inputNames.data(), 
                                encoder_inputTensors.data(), 
                                encoder_inputTensors.size(), 
                                encoder_outputNames.data(), 
                                encoder_outputNames.size());
        return out;
    }

    std::vector<Ort::Value> decoder_forward(std::vector<int64_t> in_vector, 
                                            std::vector<int64_t> dims,
                                            Ort::MemoryInfo &memory_info){
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, in_vector.data(), in_vector.size(), dims.data(), dims.size()));
    
        std::vector<const char*> inputNames {decoder.GetInputName(0, allocator)};
        std::vector<const char*> outputNames {decoder.GetOutputName(0, allocator)};

        auto out = decoder.Run(Ort::RunOptions{nullptr}, 
                                    inputNames.data(),
                                    inputTensors.data(),
                                    inputTensors.size(), 
                                    outputNames.data(),
                                    outputNames.size());

        return out;
    }

    std::vector<Ort::Value> joiner_forward(std::vector<float> projected_encoder_out, 
                                           std::vector<float> decoder_out, 
                                           std::vector<int64_t> projected_encoder_out_dims, 
                                           std::vector<int64_t> decoder_out_dims, 
                                           Ort::MemoryInfo &memory_info){
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, projected_encoder_out.data(), projected_encoder_out.size(), projected_encoder_out_dims.data(), projected_encoder_out_dims.size()));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, decoder_out.data(), decoder_out.size(), decoder_out_dims.data(), decoder_out_dims.size()));
        std::vector<const char*> inputNames = {joiner.GetInputName(0, allocator), joiner.GetInputName(1, allocator)};
        std::vector<const char*> outputNames = {joiner.GetOutputName(0, allocator)};
        
        auto out = joiner.Run(Ort::RunOptions{nullptr}, 
                                inputNames.data(), 
                                inputTensors.data(), 
                                inputTensors.size(), 
                                outputNames.data(), 
                                outputNames.size());

        return out;
    }

    std::vector<Ort::Value> joiner_encoder_proj_forward(std::vector<float> in_vector, 
                                        std::vector<int64_t> dims, 
                                        Ort::MemoryInfo &memory_info){
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, in_vector.data(), in_vector.size(), dims.data(), dims.size()));

        std::vector<const char*> inputNames {joiner_encoder_proj.GetInputName(0, allocator)};
        std::vector<const char*> outputNames {joiner_encoder_proj.GetOutputName(0, allocator)};

        auto out = joiner_encoder_proj.Run(Ort::RunOptions{nullptr}, 
                                    inputNames.data(),
                                    inputTensors.data(),
                                    inputTensors.size(), 
                                    outputNames.data(),
                                    outputNames.size());

        return out;
    }

    std::vector<Ort::Value> joiner_decoder_proj_forward(std::vector<float> in_vector, 
                                        std::vector<int64_t> dims, 
                                        Ort::MemoryInfo &memory_info){
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, in_vector.data(), in_vector.size(), dims.data(), dims.size()));

        std::vector<const char*> inputNames {joiner_decoder_proj.GetInputName(0, allocator)};
        std::vector<const char*> outputNames {joiner_decoder_proj.GetOutputName(0, allocator)};

        auto out = joiner_decoder_proj.Run(Ort::RunOptions{nullptr}, 
                                    inputNames.data(),
                                    inputTensors.data(),
                                    inputTensors.size(), 
                                    outputNames.data(),
                                    outputNames.size());

        return out;
    }

    Ort::Session load_model(const char* path){
        struct stat buffer;
        if (stat(path, &buffer) != 0){
            std::cout << "File does not exist!: " << path << std::endl;
            exit(0);
        }
        std::cout << "loading " << path << std::endl;
        Ort::Session onnx_model(env, path, session_options);
        return onnx_model;
    }

    void extract_constant_lm_parameters(){
        /*
        all_in_one contains these params. We should trace all_in_one and find 'constants_lm' nodes to extract these params
        For now, these params are set staticaly. 
        in: Ort::Session &all_in_one
        out: {blank_id, unk_id, context_size}
        should return std::vector<int32_t>
        */
        blank_id = 0;
        unk_id = 0;
        context_size = 2;
    }

    std::map<int, std::string> get_token_map(const char* token_path){
        std::ifstream inFile;
        inFile.open(token_path);
        if (inFile.fail())
                std::cerr << "Could not find token file" << std::endl;

        std::map<int, std::string> token_map;

        std::string line; 
        while (std::getline(inFile, line))
        {
            int id;
            std::string token;

            std::istringstream iss(line);
            iss >> token;
            iss >> id;

            token_map[id] = token;
        }

        return token_map;
    }

};


Model get_model(std::string exp_path, char* tokens_path){
    Model model{
        (exp_path + "/encoder_simp.onnx").c_str(),
        (exp_path + "/decoder_simp.onnx").c_str(),
        (exp_path + "/joiner_simp.onnx").c_str(),
        (exp_path + "/joiner_encoder_proj_simp.onnx").c_str(),
        (exp_path + "/joiner_decoder_proj_simp.onnx").c_str(),
        tokens_path,
    };
    model.extract_constant_lm_parameters();

    return model;
}

Model get_model(char* encoder_path,
                char* decoder_path,
                char* joiner_path,
                char* joiner_encoder_proj_path,
                char* joiner_decoder_proj_path,
                char* tokens_path){
    Model model{
        encoder_path,
        decoder_path,
        joiner_path,
        joiner_encoder_proj_path,
        joiner_decoder_proj_path,
        tokens_path,
    };
    model.extract_constant_lm_parameters();

    return model;
}


void doWarmup(Model *model, int numWarmup = 5){
    std::cout << "Warmup is started" << std::endl;

    std::vector<float> encoder_warmup_sample (500 * 80, 1.0);
    for (int i=0; i<numWarmup; i++)
        auto encoder_out = model->encoder_forward(encoder_warmup_sample,
                                            std::vector<int64_t> {500},
                                            std::vector<int64_t> {1, 500, 80},
                                            std::vector<int64_t> {1},
                                            memory_info);

    std::vector<int64_t> decoder_warmup_sample {1, 1};
    for (int i=0; i<numWarmup; i++)
        auto decoder_out = model->decoder_forward(decoder_warmup_sample, 
                                                std::vector<int64_t> {1, 2}, 
                                                memory_info);

    std::vector<float> joiner_warmup_sample1 (512, 1.0);
    std::vector<float> joiner_warmup_sample2 (512, 1.0);
    for (int i=0; i<numWarmup; i++)
        auto logits = model->joiner_forward(joiner_warmup_sample1,
                                joiner_warmup_sample2,
                                std::vector<int64_t> {1, 1, 1, 512},
                                std::vector<int64_t> {1, 1, 1, 512},
                                memory_info);

    std::vector<float> joiner_encoder_proj_warmup_sample (100 * 512, 1.0);
    for (int i=0; i<numWarmup; i++)
        auto projected_encoder_out = model->joiner_encoder_proj_forward(joiner_encoder_proj_warmup_sample, 
                                        std::vector<int64_t> {100, 512}, 
                                        memory_info);

    std::vector<float> joiner_decoder_proj_warmup_sample (512, 1.0);
    for (int i=0; i<numWarmup; i++)
        auto projected_decoder_out = model->joiner_decoder_proj_forward(joiner_decoder_proj_warmup_sample, 
                                            std::vector<int64_t> {1, 512}, 
                                            memory_info);
    std::cout << "Warmup is done" << std::endl;
}
