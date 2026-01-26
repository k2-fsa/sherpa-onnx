// sherpa-onnx/python/csrc/sentence-piece-tokenizer.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/python/csrc/sentence-piece-tokenizer.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/sentence-piece-tokenizer.h"

namespace sherpa_onnx {
void PybindSentencePieceTokenizer(py::module *m) {
  using PyClass = SentencePieceTokenizer;
  py::class_<PyClass>(*m, "SentencePieceTokenizer")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("vocab_json"), py::arg("token_scores_json"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "encode",
          [](const PyClass &self, const std::string &text,
             py::object out_type) -> py::object {
            auto builtins = py::module::import("builtins");
            py::object int_type = builtins.attr("int");
            py::object str_type = builtins.attr("str");

            if (out_type.is_none() || out_type.equal(str_type)) {
              std::vector<std::string> tokens;
              {
                py::gil_scoped_release release;
                tokens = self.EncodeTokens(text);
              }
              return py::cast(tokens);
            } else if (out_type.equal(int_type)) {
              std::vector<int32_t> ids;
              {
                py::gil_scoped_release release;
                ids = self.EncodeIds(text);
              }
              return py::cast(ids);
            } else {
              throw std::runtime_error(
                  "Invalid out_type. Must be int, str, or None.");
            }
          },
          py::arg("text"), py::arg("out_type") = py::none(),
          "Encode text. out_type can be int, str, or None. Default to str");
}

}  // namespace sherpa_onnx
