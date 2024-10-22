// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class WaveReader {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private final int sampleRate;
    private final float[] samples;

    // It supports only single channel, 16-bit wave file.
    // It will exit the program if the given file has a wrong format
    public WaveReader(String filename) {
        Object[] arr = readWaveFromFile(filename);
        samples = (float[]) arr[0];
        sampleRate = (int) arr[1];
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public float[] getSamples() {
        return samples;
    }

    private native Object[] readWaveFromFile(String filename);
}
