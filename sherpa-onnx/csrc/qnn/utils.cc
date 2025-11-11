// sherpa-onnx/csrc/qnn/utils.h
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/qnn/utils.h"

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/qnn/macros.h"

#define SHERPA_ONNX_TO_STRING(s) \
  case s:                        \
    return #s

std::string TensorTypeToString(Qnn_TensorType_t t) {
  switch (t) {
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_APP_WRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_APP_READ);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_APP_READWRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_NATIVE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_STATIC);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_NULL);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UPDATEABLE_STATIC);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UPDATEABLE_NATIVE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UPDATEABLE_APP_READ);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_OPTIONAL_APP_READ);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE);
    SHERPA_ONNX_TO_STRING(QNN_TENSOR_TYPE_UNDEFINED);
  }
  return "Unknown";
}

std::string QuantizationEncodingToString(Qnn_QuantizationEncoding_t q) {
  switch (q) {
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_BLOCK);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_VECTOR);
    SHERPA_ONNX_TO_STRING(QNN_QUANTIZATION_ENCODING_UNDEFINED);
  }
  return "Unknown";
}

std::string TensorDataTypeToString(Qnn_DataType_t t) {
  switch (t) {
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_INT_8);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_INT_16);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_INT_32);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_INT_64);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UINT_8);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UINT_16);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UINT_32);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UINT_64);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_FLOAT_16);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_FLOAT_32);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_FLOAT_64);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_SFIXED_POINT_4);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_SFIXED_POINT_8);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_SFIXED_POINT_16);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_SFIXED_POINT_32);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UFIXED_POINT_4);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UFIXED_POINT_8);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UFIXED_POINT_16);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UFIXED_POINT_32);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_BOOL_8);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_STRING);
    SHERPA_ONNX_TO_STRING(QNN_DATATYPE_UNDEFINED);
  }
  return "unknown";
}

std::string TensorMemTypeToString(Qnn_TensorMemType_t t) {
  switch (t) {
    SHERPA_ONNX_TO_STRING(QNN_TENSORMEMTYPE_RAW);
    SHERPA_ONNX_TO_STRING(QNN_TENSORMEMTYPE_MEMHANDLE);
    SHERPA_ONNX_TO_STRING(QNN_TENSORMEMTYPE_RETRIEVE_RAW);
    SHERPA_ONNX_TO_STRING(QNN_TENSORMEMTYPE_UNDEFINED);
  }
  return "Unknown";
}

#undef SHERPA_ONNX_TO_STRING

// quantized = float / scale - offset;
void FillData(Qnn_Tensor_t *t, const float *data, int32_t n) {
  float scale = t->v1.quantizeParams.scaleOffsetEncoding.scale;
  int32_t offset = t->v1.quantizeParams.scaleOffsetEncoding.offset;

  size_t bit_width = 16;
  double true_bit_width_max = pow(2, bit_width) - 1;
  double encoding_min = offset * scale;
  double encoding_max = (true_bit_width_max + offset) * scale;
  double encoding_range = encoding_max - encoding_min;

  uint16_t *out = reinterpret_cast<uint16_t *>(t->v1.clientBuf.data);

  for (size_t i = 0; i < n; ++i) {
    int32_t quantized_value =
        round(true_bit_width_max * (data[i] - encoding_min) / encoding_range);

    if (quantized_value < 0) {
      quantized_value = 0;
    } else if (quantized_value > static_cast<int32_t>(true_bit_width_max)) {
      quantized_value = static_cast<int32_t>(true_bit_width_max);
    }
    out[i] = static_cast<uint16_t>(quantized_value);
  }
}

void FillData(Qnn_Tensor_t *t, const int32_t *data, int32_t n) {
  int32_t *out = reinterpret_cast<int32_t *>(t->v1.clientBuf.data);
  std::copy(data, data + n, out);
}

void GetData(const Qnn_Tensor_t *t, float *data, int32_t n) {
  double scale = t->v1.quantizeParams.scaleOffsetEncoding.scale;
  double offset = t->v1.quantizeParams.scaleOffsetEncoding.offset;

  const uint16_t *p = reinterpret_cast<const uint16_t *>(t->v1.clientBuf.data);
  for (int32_t i = 0; i < n; ++i) {
    double quantizedValue = static_cast<double>(p[i]);
    data[i] = (quantizedValue + offset) * scale;
  }
}

static void FreeTensorV1(Qnn_Tensor_t *t) {
  free(const_cast<char *>(t->v1.name));

  delete[] t->v1.dimensions;
}

static void FreeTensorV2(Qnn_Tensor_t *t) {
  free(const_cast<char *>(t->v2.name));

  delete[] t->v2.dimensions;
  delete[] t->v2.isDynamicDimensions;
}

void FreeTensor(Qnn_Tensor_t *t) {
  if (t->version == QNN_TENSOR_VERSION_1) {
    FreeTensorV1(t);
  } else if (t->version == QNN_TENSOR_VERSION_2) {
    FreeTensorV2(t);
  } else {
    SHERPA_ONNX_LOGE("Unknown tensor version: %d", t->version);
  }
}

uint32_t GetSizeInBytes(const uint32_t *dimensions, uint32_t n,
                        Qnn_DataType_t type) {
  if (n == 0) {
    return 0;
  }

  auto count = std::accumulate(dimensions, dimensions + n, 1,
                               std::multiplies<uint32_t>());

  uint32_t b = 1;
  switch (type) {
    case QNN_DATATYPE_INT_8:
      b = 1;
      break;
    case QNN_DATATYPE_INT_16:
      b = 2;
      break;
    case QNN_DATATYPE_INT_32:
      b = 4;
      break;
    case QNN_DATATYPE_INT_64:
      b = 8;
      break;
    case QNN_DATATYPE_UINT_8:
      b = 1;
      break;
    case QNN_DATATYPE_UINT_16:
      b = 2;
      break;
    case QNN_DATATYPE_UINT_32:
      b = 4;
      break;
    case QNN_DATATYPE_UINT_64:
      b = 8;
      break;
    case QNN_DATATYPE_FLOAT_16:
      b = 2;
      break;
    case QNN_DATATYPE_FLOAT_32:
      b = 4;
      break;
    case QNN_DATATYPE_FLOAT_64:
      b = 8;
      break;
    case QNN_DATATYPE_SFIXED_POINT_8:
      b = 1;
      break;
    case QNN_DATATYPE_SFIXED_POINT_16:
      b = 2;
      break;
    case QNN_DATATYPE_SFIXED_POINT_32:
      b = 4;
      break;
    case QNN_DATATYPE_UFIXED_POINT_8:
      b = 1;
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      b = 2;
      break;
    case QNN_DATATYPE_UFIXED_POINT_32:
      b = 4;
      break;
    case QNN_DATATYPE_BOOL_8:
      b = 1;
      break;
    default:
      SHERPA_ONNX_LOGE("Unsupported data type: %s",
                       TensorDataTypeToString(type).c_str());
      break;
  }

  return count * b;
}

template <typename T>
void CopyDimensions(const T *src, uint32_t n, T **dst) {
  if (!src || n == 0) {
    *dst = nullptr;
    return;
  }

  *dst = new T[n];
  std::copy(src, src + n, *dst);
}

static void CopyQuantizeParams(const Qnn_QuantizeParams_t &src,
                               Qnn_QuantizeParams_t &dst) {  // NOLINT
  dst.encodingDefinition = src.encodingDefinition;
  dst.quantizationEncoding = src.quantizationEncoding;

  switch (src.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      dst.scaleOffsetEncoding = src.scaleOffsetEncoding;
      break;
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      // do nothing in this case
      break;
    default:
      SHERPA_ONNX_LOGE(
          "Unsupported quantizationEncoding: %s",
          QuantizationEncodingToString(src.quantizationEncoding).c_str());
  }
}

static void CopyTensorInfoV1(const Qnn_Tensor_t &src,
                             Qnn_Tensor_t &dst) {  // NOLINT
  dst.version = src.version;
  dst.v1.id = src.v1.id;
  if (src.v1.name) {
    dst.v1.name = strdup(src.v1.name);
  } else {
    dst.v1.name = strdup("");
  }

  dst.v1.type = src.v1.type;
  dst.v1.dataFormat = src.v1.dataFormat;
  dst.v1.dataType = src.v1.dataType;

  CopyQuantizeParams(src.v1.quantizeParams, dst.v1.quantizeParams);

  dst.v1.rank = src.v1.rank;

  CopyDimensions(src.v1.dimensions, src.v1.rank, &dst.v1.dimensions);

  dst.v1.memType = src.v1.memType;
  if (dst.v1.memType != QNN_TENSORMEMTYPE_RAW) {
    SHERPA_ONNX_LOGE("Unsupported mem type: %s",
                     TensorMemTypeToString(dst.v1.memType).c_str());
  } else {
    dst.v1.clientBuf.data = nullptr;
    dst.v1.clientBuf.dataSize =
        GetSizeInBytes(dst.v1.dimensions, dst.v1.rank, dst.v1.dataType);
  }
}

static void CopyTensorInfoV2(const Qnn_Tensor_t &src,
                             Qnn_Tensor_t &dst) {  // NOLINT
  dst.version = src.version;
  dst.v2.id = src.v2.id;
  if (src.v2.name) {
    dst.v2.name = strdup(src.v2.name);
  } else {
    dst.v2.name = strdup("");
  }

  dst.v2.type = src.v2.type;
  dst.v2.dataFormat = src.v2.dataFormat;
  dst.v2.dataType = src.v2.dataType;

  CopyQuantizeParams(src.v2.quantizeParams, dst.v2.quantizeParams);

  dst.v2.rank = src.v2.rank;

  CopyDimensions(src.v2.dimensions, src.v2.rank, &dst.v2.dimensions);

  dst.v2.memType = src.v2.memType;
  if (dst.v2.memType != QNN_TENSORMEMTYPE_RAW) {
    SHERPA_ONNX_LOGE("Unsupported mem type: %s",
                     TensorMemTypeToString(dst.v2.memType).c_str());
  } else {
    dst.v2.clientBuf.data = nullptr;
    dst.v2.clientBuf.dataSize =
        GetSizeInBytes(dst.v2.dimensions, dst.v2.rank, dst.v2.dataType);
  }

  CopyDimensions(src.v2.isDynamicDimensions, src.v2.rank,
                 &dst.v2.isDynamicDimensions);

  dst.v2.sparseParams.type = src.v2.sparseParams.type;
  dst.v2.sparseParams.hybridCoo.numSpecifiedElements =
      src.v2.sparseParams.hybridCoo.numSpecifiedElements;
  dst.v2.sparseParams.hybridCoo.numSparseDimensions =
      src.v2.sparseParams.hybridCoo.numSparseDimensions;
  dst.v2.isProduced = src.v2.isProduced;
}

void CopyTensorInfo(const Qnn_Tensor_t &src, Qnn_Tensor_t &dst) {  // NOLINT
  if (src.version == QNN_TENSOR_VERSION_1) {
    CopyTensorInfoV1(src, dst);
  } else if (src.version == QNN_TENSOR_VERSION_2) {
    CopyTensorInfoV2(src, dst);
  } else {
    SHERPA_ONNX_LOGE("Unknown tensor version: %d", dst.version);
  }
}

void LogCallback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp,
                 va_list args) {
  std::string s;
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      s = "ERROR";
      break;
    case QNN_LOG_LEVEL_WARN:
      s = "WARN";
      break;
    case QNN_LOG_LEVEL_INFO:
      s = "INFO";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      s = "DEBUG";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      s = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      s = "UNKNOWN";
      break;
  }

  double ms = timestamp / 1000000.0;
  fprintf(stdout, "%8.1fms [%-7s] ", ms, s.c_str());
  vfprintf(stdout, fmt, args);
}

void PrintTensor(Qnn_TensorV2_t t) {
  std::ostringstream os;
  os << "  id: " << t.id << "\n";
  os << "  name: " << t.name << "\n";
  os << "  type: " << TensorTypeToString(t.type) << "\n";
  os << "  data format: " << t.dataFormat << "\n";
  os << "  data type: " << TensorDataTypeToString(t.dataType) << "\n";
  os << "  quantize info: \n";
  auto qp = t.quantizeParams;
  os << "    encodingDefinition: " << std::hex << "0x" << qp.encodingDefinition
     << std::dec << "\n";
  os << "    quantizationEncoding: "
     << QuantizationEncodingToString(qp.quantizationEncoding) << "\n";
  if (qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    Qnn_ScaleOffset_t s = qp.scaleOffsetEncoding;
    os << "     scale: " << s.scale << "\n";
    os << "     offset: " << s.offset << "\n";
  }
  os << "  rank: " << t.rank << "\n";
  os << "  dimensions: ";
  for (int32_t i = 0; i < t.rank; ++i) {
    os << t.dimensions[i] << ", ";
    if (i + 1 == t.rank) {
      os << "\n";
    }
  }
  os << "  memType: " << TensorMemTypeToString(t.memType) << "\n";
  if (t.memType == QNN_TENSORMEMTYPE_RAW) {
    os << " memType raw data size: " << t.clientBuf.dataSize << "\n";
  }
  os << "  isDynamicDimensions: "
     << ((t.isDynamicDimensions != nullptr) ? "True" : "False") << "\n";
  os << "  isProduced: " << static_cast<int32_t>(t.isProduced) << "\n";

  SHERPA_ONNX_LOGE("%s", os.str().c_str());
}

static bool CopyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t *src,
                             GraphInfo *dst) {
  if (src->graphName) {
    dst->graph_name = strdup(src->graphName);
  } else {
    dst->graph_name = strdup("");
  }

  dst->input_tensors = nullptr;
  dst->num_input_tensors = 0;

  if (src->graphInputs) {
    dst->input_tensors = reinterpret_cast<Qnn_Tensor_t *>(
        calloc(src->numGraphInputs, sizeof(Qnn_Tensor_t)));

    for (uint32_t i = 0; i < src->numGraphInputs; ++i) {
      dst->input_tensors[i] = QNN_TENSOR_INIT;

      CopyTensorInfo(src->graphInputs[i], dst->input_tensors[i]);
    }

    dst->num_input_tensors = src->numGraphInputs;
  }

  dst->output_tensors = nullptr;
  dst->num_output_tensors = 0;

  if (src->graphOutputs) {
    dst->output_tensors = reinterpret_cast<Qnn_Tensor_t *>(
        calloc(src->numGraphOutputs, sizeof(Qnn_Tensor_t)));

    for (uint32_t i = 0; i < src->numGraphOutputs; ++i) {
      dst->output_tensors[i] = QNN_TENSOR_INIT;

      CopyTensorInfo(src->graphOutputs[i], dst->output_tensors[i]);
    }

    dst->num_output_tensors = src->numGraphOutputs;
  }

  return true;
}

static bool CopyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphs_input,
                           uint32_t num_graphs,
                           GraphInfo **&graphs_info) {  // NOLINT
  SHERPA_ONNX_LOGE("version: %d", (int)graphs_input[0].version);

  // remember to free graphs_info
  graphs_info =
      reinterpret_cast<GraphInfo **>(calloc(num_graphs, sizeof(GraphInfo *)));

  GraphInfo *graph_info_arr =
      reinterpret_cast<GraphInfo *>(calloc(num_graphs, sizeof(GraphInfo)));

  if (!graphs_info || !graph_info_arr) {
    SHERPA_ONNX_LOGE("Failure to allocate memory for *graphInfo");
    return false;
  }

  for (uint32_t i = 0; i < num_graphs; ++i) {
    switch (graphs_input[i].version) {
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
        SHERPA_ONNX_LOGE("Unsupported version: %d",
                         static_cast<int32_t>(graphs_input[i].version));
        return false;

      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
        SHERPA_ONNX_LOGE("Unsupported version: %d",
                         static_cast<int32_t>(graphs_input[i].version));
        return false;

      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
        bool ok =
            CopyGraphsInfoV3(&graphs_input[i].graphInfoV3, &graph_info_arr[i]);
        if (!ok) {
          SHERPA_ONNX_LOGE("Failed to copy graphs info v3");
        }
        graphs_info[i] = graph_info_arr + i;

        break;
      }

      default:
        SHERPA_ONNX_LOGE("Unsupported version: %d",
                         static_cast<int32_t>(graphs_input[i].version));
        return false;
    }
  }

  return true;
}

bool CopyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binary_info,
                              GraphInfo **&graphs_info,  // NOLINT
                              uint32_t &graphs_count) {  // NOLINT
  graphs_count = 0;

  switch (binary_info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
      SHERPA_ONNX_LOGE("Unsupported binary context version: %d",
                       binary_info->version);
      return false;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
      SHERPA_ONNX_LOGE("Unsupported binary context version: %d",
                       binary_info->version);
      return false;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
      bool ok = CopyGraphsInfo(binary_info->contextBinaryInfoV3.graphs,
                               binary_info->contextBinaryInfoV3.numGraphs,
                               graphs_info);

      if (!ok) {
        SHERPA_ONNX_LOGE("Failed while copying graphs Info v3.");
        return false;
      }
      graphs_count = binary_info->contextBinaryInfoV3.numGraphs;
      return true;
    }
    default: {
      SHERPA_ONNX_LOGE("Unsupported binary context version: %d",
                       binary_info->version);
      return false;
    }
  }
}
