// sherpa-onnx/csrc/fast-clustering-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
std::string FastClusteringConfig::ToString() const {
  std::ostringstream os;

  os << "FastClusteringConfig(";
  os << "num_clusters=" << num_clusters << ", ";
  os << "threshold=" << threshold << ", ";
  os << "compute_confidence=" << (compute_confidence ? "True" : "False") << ")";

  return os.str();
}

void FastClusteringConfig::Register(ParseOptions *po) {
  po->Register(
      "num-clusters", &num_clusters,
      "Number of cluster. If greater than 0, then cluster threshold is "
      "ignored. Please provide it if you know the actual number of "
      "clusters in advance.");

  po->Register("cluster-threshold", &threshold,
               "If num_clusters is not specified, then it specifies the "
               "distance threshold for clustering. smaller value -> more "
               "clusters. larger value -> fewer clusters");

  po->Register("compute-confidence", &compute_confidence,
               "True to compute confidence value for each clustered "
               "embedding and surface for every diarization segment.");
}

bool FastClusteringConfig::Validate() const {
  if (num_clusters < 1 && threshold < 0) {
    SHERPA_ONNX_LOGE("Please provide either num_clusters or threshold");
    return false;
  }

  return true;
}

}  // namespace sherpa_onnx
