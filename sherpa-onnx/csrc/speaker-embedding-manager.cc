// sherpa-onnx/csrc/speaker-embedding-manager.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/speaker-embedding-manager.h"

#include <algorithm>
#include <unordered_map>

#include "Eigen/Dense"

namespace sherpa_onnx {

using FloatMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class SpeakerEmbeddingManager::Impl {
 public:
  explicit Impl(int32_t dim) : dim_(dim) {}

  bool Add(const std::string &name, const float *p) {
    if (name2row_.count(name)) {
      // a speaker with the same name already exists
      return false;
    }

    embedding_matrix_.conservativeResize(embedding_matrix_.rows() + 1, dim_);

    std::copy(p, p + dim_, &embedding_matrix_.bottomRows(1)(0, 0));

    embedding_matrix_.bottomRows(1).normalize();  // inplace

    name2row_[name] = embedding_matrix_.rows() - 1;
    row2name_[embedding_matrix_.rows() - 1] = name;

    return true;
  }

  bool Remove(const std::string &name) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    int32_t num_rows = embedding_matrix_.rows();

    if (row_idx < num_rows - 1) {
      embedding_matrix_.block(row_idx, 0, num_rows - -1 - row_idx, dim_) =
          embedding_matrix_.bottomRows(num_rows - 1 - row_idx);
    }

    embedding_matrix_.conservativeResize(num_rows - 1, dim_);
    for (auto &p : name2row_) {
      if (p.second > row_idx) {
        p.second -= 1;
        row2name_[p.second] = p.first;
      }
    }

    name2row_.erase(name);
    row2name_.erase(num_rows - 1);

    return true;
  }

  std::string Search(const float *p, float threshold) {
    if (embedding_matrix_.rows() == 0) {
      return {};
    }

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    Eigen::VectorXf scores = embedding_matrix_ * v;

    Eigen::VectorXf::Index max_index;
    float max_score = scores.maxCoeff(&max_index);
    if (max_score < threshold) {
      return {};
    }

    return row2name_.at(max_index);
  }

  bool Verify(const std::string &name, const float *p, float threshold) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    float score = embedding_matrix_.row(row_idx) * v;

    if (score < threshold) {
      return false;
    }

    return true;
  }

  int32_t NumSpeakers() const { return embedding_matrix_.rows(); }

 private:
  int32_t dim_;
  FloatMatrix embedding_matrix_;
  std::unordered_map<std::string, int32_t> name2row_;
  std::unordered_map<int32_t, std::string> row2name_;
};

SpeakerEmbeddingManager::SpeakerEmbeddingManager(int32_t dim)
    : impl_(std::make_unique<Impl>(dim)) {}

SpeakerEmbeddingManager::~SpeakerEmbeddingManager() = default;

bool SpeakerEmbeddingManager::Add(const std::string &name,
                                  const float *p) const {
  return impl_->Add(name, p);
}

bool SpeakerEmbeddingManager::Remove(const std::string &name) const {
  return impl_->Remove(name);
}

std::string SpeakerEmbeddingManager::Search(const float *p,
                                            float threshold) const {
  return impl_->Search(p, threshold);
}

bool SpeakerEmbeddingManager::Verify(const std::string &name, const float *p,
                                     float threshold) const {
  return impl_->Verify(name, p, threshold);
}

int32_t SpeakerEmbeddingManager::NumSpeakers() const {
  return impl_->NumSpeakers();
}

}  // namespace sherpa_onnx
