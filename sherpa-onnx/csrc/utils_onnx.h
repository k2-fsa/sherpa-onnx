#include <iostream>
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
const auto& api = Ort::GetApi();
OrtTensorRTProviderOptionsV2* tensorrt_options;
Ort::SessionOptions session_options;
Ort::AllocatorWithDefaultOptions allocator;
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


std::vector<float> ortVal2Vector(Ort::Value &tensor, int tensor_length){
    /**
     * convert ort tensor to vector
     */
    float* floatarr = tensor.GetTensorMutableData<float>();
    std::vector<float> vector {floatarr, floatarr + tensor_length};
    return vector;
}


void print_onnx_forward_output(std::vector<Ort::Value> &output_tensors, int num){
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    for (int i = 0; i < num; i++)
        printf("[%d] =  %f\n", i, floatarr[i]);
}


void print_shape_of_ort_val(std::vector<Ort::Value> &tensor){
    auto out_shape = tensor.front().GetTensorTypeAndShapeInfo().GetShape();
    auto out_size = out_shape.size();
    std::cout << "(";
    for (int i=0; i<out_size; i++){
        std::cout << out_shape[i];
        if (i < out_size-1)
        std::cout << ",";
    }
    std::cout << ")" << std::endl;
}


void print_model_info(Ort::Session &session, std::string title){
    std::cout << "=== Printing '" << title  << "' model ===" << std::endl;
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    printf("Number of inputs = %zu\n", num_input_nodes);

    char* output_name = session.GetOutputName(0, allocator);
    printf("output name: %s\n", output_name);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (size_t j = 0; j < input_node_dims.size(); j++)
        printf("Input %d : dim %zu=%jd\n", i, j, input_node_dims[j]);
    }
    std::cout << "=======================================" << std::endl;
}
