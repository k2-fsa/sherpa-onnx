// sherpa-onnx/python/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/circular-buffer.h"

#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"

namespace sherpa_onnx {

static constexpr const char *kCircularBufferInitDoc = R"doc(
Constructor for CircularBuffer.

Args:
  capacity:
    The maximum number of samples the buffer can hold.
)doc";

static constexpr const char *kCircularBufferPushDoc = R"doc(
Push samples into the buffer.

Args:
  samples:
    A 1-D float32 array of samples to push.
)doc";

static constexpr const char *kCircularBufferGetDoc = R"doc(
Get n samples starting from the given index.

Args:
  start_index:
    The start index in the buffer.
  n:
    Number of samples to retrieve.

Returns:
  A 1-D float32 numpy array of n samples.
)doc";

static constexpr const char *kCircularBufferPopDoc = R"doc(
Remove n samples from the front of the buffer.

Args:
  n:
    Number of samples to remove.
)doc";

static constexpr const char *kCircularBufferResetDoc = R"doc(
Reset the buffer, discarding all data.
)doc";

static constexpr const char *kCircularBufferSizeDoc = R"doc(
Return the number of samples currently in the buffer.
)doc";

static constexpr const char *kCircularBufferHeadDoc = R"doc(
Return the index of the oldest sample in the buffer.
)doc";

static constexpr const char *kCircularBufferTailDoc = R"doc(
Return the index of the next sample to be added.
)doc";

void PybindCircularBuffer(py::module *m) {
  using PyClass = CircularBuffer;
  py::class_<PyClass>(*m, "CircularBuffer")
      .def(py::init<int32_t>(), py::arg("capacity"),
           kCircularBufferInitDoc)
      .def(
          "push",
          [](PyClass &self, const std::vector<float> &samples) {
            self.Push(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>(),
          kCircularBufferPushDoc)
      .def("get", &PyClass::Get, py::arg("start_index"), py::arg("n"),
           py::call_guard<py::gil_scoped_release>(), kCircularBufferGetDoc)
      .def("pop", &PyClass::Pop, py::arg("n"),
           py::call_guard<py::gil_scoped_release>(), kCircularBufferPopDoc)
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>(),
           kCircularBufferResetDoc)
      .def_property_readonly("size", &PyClass::Size, kCircularBufferSizeDoc)
      .def_property_readonly("head", &PyClass::Head, kCircularBufferHeadDoc)
      .def_property_readonly("tail", &PyClass::Tail, kCircularBufferTailDoc);
}

}  // namespace sherpa_onnx
