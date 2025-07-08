// sherpa-onnx/python/csrc/offline-source-separation-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation.h"

#include <algorithm>
#include <string>

#include "sherpa-onnx/python/csrc/offline-source-separation-model-config.h"
#include "sherpa-onnx/python/csrc/offline-source-separation.h"

#define C_CONTIGUOUS py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_

namespace sherpa_onnx {

static void PybindOfflineSourceSeparationConfig(py::module *m) {
  PybindOfflineSourceSeparationModelConfig(m);

  using PyClass = OfflineSourceSeparationConfig;
  py::class_<PyClass>(*m, "OfflineSourceSeparationConfig")
      .def(py::init<const OfflineSourceSeparationModelConfig &>(),
           py::arg("model") = OfflineSourceSeparationModelConfig{})
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindMultiChannelSamples(py::module *m) {
  using PyClass = MultiChannelSamples;

  py::class_<PyClass>(*m, "MultiChannelSamples")
      .def_property_readonly("data", [](PyClass &self) -> py::object {
        // if data is not empty, return a float array of
        // shape (num_channels, num_samples)
        int32_t num_channels = self.data.size();
        if (num_channels == 0) {
          return py::none();
        }

        int32_t num_samples = self.data[0].size();
        if (num_samples == 0) {
          return py::none();
        }

        py::array_t<float> ans({num_channels, num_samples});

        py::buffer_info buf = ans.request();
        auto p = static_cast<float *>(buf.ptr);

        for (int32_t i = 0; i != num_channels; ++i) {
          std::copy(self.data[i].begin(), self.data[i].end(),
                    p + i * num_samples);
        }

        return ans;
      });
}

static void PybindOfflineSourceSeparationOutput(py::module *m) {
  using PyClass = OfflineSourceSeparationOutput;
  py::class_<PyClass>(*m, "OfflineSourceSeparationOutput")
      .def_property_readonly(
          "sample_rate", [](const PyClass &self) { return self.sample_rate; })
      .def_property_readonly("stems",
                             [](const PyClass &self) { return self.stems; });
}

void PybindOfflineSourceSeparation(py::module *m) {
  PybindOfflineSourceSeparationConfig(m);
  PybindOfflineSourceSeparationOutput(m);

  PybindMultiChannelSamples(m);

  using PyClass = OfflineSourceSeparation;
  py::class_<PyClass>(*m, "OfflineSourceSeparation")
      .def(py::init<const OfflineSourceSeparationConfig &>(),
           py::arg("config") = OfflineSourceSeparationConfig{})
      .def(
          "process",
          [](const PyClass &self, int32_t sample_rate,
             const py::array_t<float> &samples) {
            if (!(C_CONTIGUOUS == (samples.flags() & C_CONTIGUOUS))) {
              throw py::value_error(
                  "input samples should be contiguous. Please use "
                  "np.ascontiguousarray(samples)");
            }

            int num_dim = samples.ndim();
            if (samples.ndim() != 2) {
              std::ostringstream os;
              os << "Expect an array of 2 dimensions [num_channels x "
                    "num_samples]. "
                    "Given dim: "
                 << num_dim << "\n";
              throw py::value_error(os.str());
            }

            // if num_samples is less than 10, it is very likely the user
            // has swapped num_channels and num_samples.
            if (samples.shape(1) < 10) {
              std::ostringstream os;
              os << "Expect an array of 2 dimensions [num_channels x "
                    "num_samples]. "
                    "Given ["
                 << samples.shape(0) << " x " << samples.shape(1) << "]"
                 << "\n";
              throw py::value_error(os.str());
            }

            int32_t num_channels = samples.shape(0);
            int32_t num_samples = samples.shape(1);
            const float *p = samples.data();

            OfflineSourceSeparationInput input;

            input.samples.data.resize(num_channels);
            input.sample_rate = sample_rate;

            for (int32_t i = 0; i != num_channels; ++i) {
              input.samples.data[i] = {p + i * num_samples,
                                       p + (i + 1) * num_samples};
            }

            pybind11::gil_scoped_release release;

            return self.Process(input);
          },
          py::arg("sample_rate"), py::arg("samples"),
          "samples is of shape (num_channels, num-samples) with dtype "
          "np.float32");
}

}  // namespace sherpa_onnx
