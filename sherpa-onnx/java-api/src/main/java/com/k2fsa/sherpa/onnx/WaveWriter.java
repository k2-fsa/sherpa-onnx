// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class WaveWriter {
    public WaveWriter() {
    }

    public static boolean write(String filename, float[] samples, int sampleRate) {
        WaveWriter w = new WaveWriter();
        return w.writeWaveToFile(filename, samples, sampleRate);
    }

    private native boolean writeWaveToFile(String filename, float[] samples, int sampleRate);
}
