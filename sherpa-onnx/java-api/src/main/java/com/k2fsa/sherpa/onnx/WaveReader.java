// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class WaveReader {
    private WaveData data;

    // It supports only single channel, 16-bit wave file.
    // It will exit the program if the given file has a wrong format
    public WaveReader(String filename) {
        LibraryLoader.maybeLoad();
        this.data = readWaveFromFile(filename);
    }

    public int getSampleRate() {
        return this.data.getSampleRate();
    }

    public float[] getSamples() {
        return this.data.getSamples();
    }

    private native WaveData readWaveFromFile(String filename);
}
