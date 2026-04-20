// Copyright (c) 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

import java.util.Arrays;

public class WaveData {
    private final float[] samples;
    private final int sampleRate;

    public WaveData(float[] samples, int sampleRate) {
        this.samples = samples;
        this.sampleRate = sampleRate;
    }

    public float[] getSamples() {
        return samples;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        WaveData other = (WaveData) obj;
        return sampleRate == other.sampleRate && Arrays.equals(samples, other.samples);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(samples);
        result = 31 * result + sampleRate;
        return result;
    }
}

