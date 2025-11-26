// sherpa-onnx/csrc/rknn/context-blocking-queue-rknn.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/context-blocking-queue-rknn.h"

#include <condition_variable>
#include <mutex>
#include <queue>

#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/rknn/utils.h"

namespace sherpa_onnx {

class ContextBlockingQueueRknn::Impl {
 public:
  Impl(rknn_context context, int32_t num_threads, int32_t capacity) {
    for (int32_t i = 0; i < capacity; ++i) {
      rknn_context bak = 0;
      auto ret = rknn_dup_context(&context, &bak);
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to duplicate context");

      SetCoreMask(bak, num_threads);
      queue_.push(bak);
    }
  }
  rknn_context Take() {
    std::unique_lock<std::mutex> lock(mutex_);

    cv_.wait(lock, [&] { return stopped_ || !queue_.empty(); });

    if (stopped_ && queue_.empty()) {
      return 0;
    }

    rknn_context ctx = queue_.front();
    queue_.pop();
    return ctx;
  }

  void Put(rknn_context ctx) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        rknn_destroy(ctx);
        return;
      }
      queue_.push(ctx);
    }
    cv_.notify_one();
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
    Cleanup();
  }

 private:
  void Cleanup() {
    while (!queue_.empty()) {
      rknn_destroy(queue_.front());
      queue_.pop();
    }
  }

  std::queue<rknn_context> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_ = false;
};

ContextBlockingQueueRknn::ContextBlockingQueueRknn(rknn_context context,
                                                   int32_t num_threads,
                                                   int32_t capacity /*= 20*/)
    : impl_(std::make_unique<Impl>(context, num_threads, capacity)) {}

ContextBlockingQueueRknn::~ContextBlockingQueueRknn() = default;

rknn_context ContextBlockingQueueRknn::Take() { return impl_->Take(); }

void ContextBlockingQueueRknn::Put(rknn_context context) {
  impl_->Put(context);
}

}  // namespace sherpa_onnx
