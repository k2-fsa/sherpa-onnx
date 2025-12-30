// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

@FunctionalInterface
public interface OfflineSpeakerDiarizationCallback {
    Integer invoke(int numProcessedChunks, int numTotalCunks, long arg);
}
