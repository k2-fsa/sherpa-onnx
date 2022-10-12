#include <vector>
#include <iostream>
#include <algorithm>
#include <time.h>

#include "models.h"
#include "utils.h"


std::vector<float> getEncoderCol(Ort::Value &tensor, int start, int length){
    float* floatarr = tensor.GetTensorMutableData<float>();
    std::vector<float> vector {floatarr + start, floatarr + length};
    return vector;
}


/**
 * Assume batch size = 1
 */
std::vector<int64_t> BuildDecoderInput(const std::vector<std::vector<int32_t>> &hyps,
                              std::vector<int64_t> &decoder_input) {

    int32_t context_size = decoder_input.size();
    int32_t hyps_length = hyps[0].size();
    for (int i=0; i < context_size; i++)
        decoder_input[i] = hyps[0][hyps_length-context_size+i];

    return decoder_input;
}


std::vector<std::vector<int32_t>> GreedySearch(
                        Model *model,  // NOLINT
                        std::vector<Ort::Value> *encoder_out){
    Ort::Value &encoder_out_tensor = encoder_out->at(0);
    int encoder_out_dim1 = encoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[1];
    int encoder_out_dim2 = encoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[2];
    auto encoder_out_vector = ortVal2Vector(encoder_out_tensor, encoder_out_dim1 * encoder_out_dim2);

    // # === Greedy Search === #
    int32_t batch_size = 1;
    std::vector<int32_t> blanks(model->context_size, model->blank_id);
    std::vector<std::vector<int32_t>> hyps(batch_size, blanks);
    std::vector<int64_t> decoder_input(model->context_size, model->blank_id);

    auto decoder_out = model->decoder_forward(decoder_input, 
                                            std::vector<int64_t> {batch_size, model->context_size}, 
                                            memory_info);

    Ort::Value &decoder_out_tensor = decoder_out[0];
    int decoder_out_dim = decoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[2];
    auto decoder_out_vector = ortVal2Vector(decoder_out_tensor, decoder_out_dim);

    decoder_out = model->joiner_decoder_proj_forward(decoder_out_vector, 
                                        std::vector<int64_t> {1, decoder_out_dim}, 
                                        memory_info);
    Ort::Value &projected_decoder_out_tensor = decoder_out[0];
    auto projected_decoder_out_dim = projected_decoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[1];
    auto projected_decoder_out_vector = ortVal2Vector(projected_decoder_out_tensor, projected_decoder_out_dim);

    auto projected_encoder_out = model->joiner_encoder_proj_forward(encoder_out_vector, 
                                        std::vector<int64_t> {encoder_out_dim1, encoder_out_dim2}, 
                                        memory_info);
    Ort::Value &projected_encoder_out_tensor = projected_encoder_out[0];
    int projected_encoder_out_dim1 = projected_encoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[0];
    int projected_encoder_out_dim2 = projected_encoder_out_tensor.GetTensorTypeAndShapeInfo().GetShape()[1];
    auto projected_encoder_out_vector = ortVal2Vector(projected_encoder_out_tensor, projected_encoder_out_dim1 * projected_encoder_out_dim2);

    int32_t offset = 0;
    for (int i=0; i< projected_encoder_out_dim1; i++){
        int32_t cur_batch_size = 1;
        int32_t start = offset;
        int32_t end = start + cur_batch_size;
        offset = end;
        
        auto cur_encoder_out = getEncoderCol(projected_encoder_out_tensor, start * projected_encoder_out_dim2, end * projected_encoder_out_dim2);

        auto logits = model->joiner_forward(cur_encoder_out,
                                            projected_decoder_out_vector,
                                            std::vector<int64_t> {1, projected_encoder_out_dim2},
                                            std::vector<int64_t> {1, projected_decoder_out_dim},
                                            memory_info);

        Ort::Value &logits_tensor = logits[0];
        int logits_dim = logits_tensor.GetTensorTypeAndShapeInfo().GetShape()[1];
        auto logits_vector = ortVal2Vector(logits_tensor, logits_dim);
        
        int max_indices = static_cast<int>(std::distance(logits_vector.begin(), std::max_element(logits_vector.begin(), logits_vector.end())));
        bool emitted = false;

        for (int32_t k = 0; k != cur_batch_size; ++k) {
            auto index = max_indices;
            if (index != model->blank_id && index != model->unk_id) {
                emitted = true;
                hyps[k].push_back(index);
            }
        }

        if (emitted) {
            decoder_input = BuildDecoderInput(hyps, decoder_input);

            decoder_out = model->decoder_forward(decoder_input, 
                                        std::vector<int64_t> {batch_size, model->context_size}, 
                                        memory_info);

            decoder_out_dim = decoder_out[0].GetTensorTypeAndShapeInfo().GetShape()[2];
            decoder_out_vector = ortVal2Vector(decoder_out[0], decoder_out_dim);

            decoder_out = model->joiner_decoder_proj_forward(decoder_out_vector, 
                                                std::vector<int64_t> {1, decoder_out_dim}, 
                                                memory_info);
            
            projected_decoder_out_dim = decoder_out[0].GetTensorTypeAndShapeInfo().GetShape()[1];
            projected_decoder_out_vector = ortVal2Vector(decoder_out[0], projected_decoder_out_dim);
        }
    }

    return hyps;
}

