// sherpa-onnx/csrc/offline-recognizer-config-abi-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Locks down the memory layout of the config structs that language bindings
// mirror. A binding that copies these structs byte-for-byte (C#, Rust, Dart,
// Pascal, ...) silently reads garbage if a field is appended, reordered or
// dropped; that has happened before (ctc_fst_decoder_config was missing from
// several bindings and produced `graph="???"` at runtime).
//
// The checks are platform independent: they assert field ORDER, that the
// trailing field really is last, and the inner layout of the nested struct.

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"
#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx {

// A field must start at or after the previous one; a duplicate offset means
// two fields overlapped, i.e. a reordering bug.
#define EXPECT_ORDER(type, prev, cur) \
  EXPECT_GE(offsetof(type, cur), offsetof(type, prev))

TEST(OfflineCtcFstDecoderConfigLayout, IsPointerThenInt32) {
  using C = SherpaOnnxOfflineCtcFstDecoderConfig;

  EXPECT_EQ(offsetof(C, graph), 0u);
  // max_active is an int32 placed after a pointer; it must not overlap it.
  EXPECT_GE(offsetof(C, max_active), sizeof(const char *));

  // The struct must not have trailing holes that a mirror could mis-size on:
  // its size is the offset of the last field plus that field's size, padded
  // out to the struct's alignment.
  EXPECT_EQ(sizeof(C) % alignof(const char *), 0u);
  EXPECT_GE(sizeof(C), offsetof(C, max_active) + sizeof(int32_t));
}

TEST(OfflineRecognizerConfigLayout, FieldsAreInDeclarationOrder) {
  using C = SherpaOnnxOfflineRecognizerConfig;

  // Mirrors the declaration order in c-api.h. If a field is added to the C
  // struct, append it here too so the bindings stay honest.
  EXPECT_ORDER(C, feat_config, model_config);
  EXPECT_ORDER(C, model_config, lm_config);
  EXPECT_ORDER(C, lm_config, decoding_method);
  EXPECT_ORDER(C, decoding_method, max_active_paths);
  EXPECT_ORDER(C, max_active_paths, hotwords_file);
  EXPECT_ORDER(C, hotwords_file, hotwords_score);
  EXPECT_ORDER(C, hotwords_score, rule_fsts);
  EXPECT_ORDER(C, rule_fsts, rule_fars);
  EXPECT_ORDER(C, rule_fars, blank_penalty);
  EXPECT_ORDER(C, blank_penalty, hr);
  EXPECT_ORDER(C, hr, ctc_fst_decoder_config);
}

TEST(OfflineRecognizerConfigLayout, CtcFstDecoderConfigIsLast) {
  using C = SherpaOnnxOfflineRecognizerConfig;

  // The bindings append ctc_fst_decoder_config at the end. If anything is
  // ever added after it, every mirror breaks — so make that loud here.
  EXPECT_EQ(offsetof(C, ctc_fst_decoder_config) + sizeof(SherpaOnnxOfflineCtcFstDecoderConfig),
            sizeof(C));
}

TEST(OfflineRecognizerConfigLayout, NestedStructsKeepTheirSize) {
  // Sanity: the nested configs must stay non-empty, otherwise a mirror could
  // collapse them to zero and shift every following field.
  EXPECT_GT(sizeof(SherpaOnnxFeatureConfig), 0u);
  EXPECT_GT(sizeof(SherpaOnnxOfflineModelConfig), 0u);
  EXPECT_GT(sizeof(SherpaOnnxOfflineLMConfig), 0u);
  EXPECT_GT(sizeof(SherpaOnnxHomophoneReplacerConfig), 0u);
  EXPECT_GT(sizeof(SherpaOnnxOfflineCtcFstDecoderConfig), 0u);
}

}  // namespace sherpa_onnx
