// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

@FunctionalInterface
public interface OfflineTtsCallback {
    /**
     * @param samples audio chunk
     * @return 1 to continue, 0 to stop
     */
    Integer invoke(float[] samples);
}
