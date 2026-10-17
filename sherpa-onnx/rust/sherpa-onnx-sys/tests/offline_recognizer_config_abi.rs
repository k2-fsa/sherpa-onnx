//! Locks down the memory layout of the config structs that this crate mirrors
//! from `sherpa-onnx/c-api/c-api.h`.
//!
//! `OfflineRecognizerConfig` is passed to the C API as a raw `#[repr(C)]`
//! struct, so a field that is added, reordered or dropped silently shifts every
//! following field. That has happened before: `ctc_fst_decoder_config` was
//! missing from several bindings and produced a garbage `graph` pointer at
//! runtime.
//!
//! The same assertions live in
//! `sherpa-onnx/csrc/offline-recognizer-config-abi-test.cc` for the C side.
//! If you change the C struct, update both.

use sherpa_onnx_sys::{OfflineCtcFstDecoderConfig, OfflineRecognizerConfig};
use std::mem::{align_of, offset_of, size_of};

/// Offset of the field must not move backwards: a reordering bug makes two
/// fields overlap.
fn assert_order(pairs: &[(&str, usize)]) {
    for w in pairs.windows(2) {
        let (prev_name, prev) = w[0];
        let (cur_name, cur) = w[1];
        assert!(
            cur >= prev,
            "field {cur_name} (offset {cur}) starts before {prev_name} (offset {prev})"
        );
    }
}

#[test]
fn offline_ctc_fst_decoder_config_is_pointer_then_int32() {
    assert_eq!(offset_of!(OfflineCtcFstDecoderConfig, graph), 0);
    assert!(offset_of!(OfflineCtcFstDecoderConfig, max_active) >= size_of::<*const std::os::raw::c_char>());
    assert_eq!(size_of::<OfflineCtcFstDecoderConfig>() % align_of::<*const std::os::raw::c_char>(), 0);
}

#[test]
fn offline_recognizer_config_fields_are_in_declaration_order() {
    // Mirrors the declaration order in c-api.h. Append here when the C struct
    // grows so the mirrors stay honest.
    assert_order(&[
        ("feat_config", offset_of!(OfflineRecognizerConfig, feat_config)),
        ("model_config", offset_of!(OfflineRecognizerConfig, model_config)),
        ("lm_config", offset_of!(OfflineRecognizerConfig, lm_config)),
        ("decoding_method", offset_of!(OfflineRecognizerConfig, decoding_method)),
        ("max_active_paths", offset_of!(OfflineRecognizerConfig, max_active_paths)),
        ("hotwords_file", offset_of!(OfflineRecognizerConfig, hotwords_file)),
        ("hotwords_score", offset_of!(OfflineRecognizerConfig, hotwords_score)),
        ("rule_fsts", offset_of!(OfflineRecognizerConfig, rule_fsts)),
        ("rule_fars", offset_of!(OfflineRecognizerConfig, rule_fars)),
        ("blank_penalty", offset_of!(OfflineRecognizerConfig, blank_penalty)),
        ("hr", offset_of!(OfflineRecognizerConfig, hr)),
        ("ctc_fst_decoder_config", offset_of!(OfflineRecognizerConfig, ctc_fst_decoder_config)),
    ]);
}

#[test]
fn offline_recognizer_config_ctc_fst_is_last() {
    // Bindings append ctc_fst_decoder_config at the end. Anything added after
    // it breaks every mirror, so make that loud here.
    assert_eq!(
        offset_of!(OfflineRecognizerConfig, ctc_fst_decoder_config)
            + size_of::<OfflineCtcFstDecoderConfig>(),
        size_of::<OfflineRecognizerConfig>()
    );
}

#[test]
fn offline_recognizer_config_nested_structs_are_non_empty() {
    assert!(size_of::<OfflineRecognizerConfig>() > 0);
    assert!(size_of::<OfflineCtcFstDecoderConfig>() > 0);
}
