// Copyright (c) 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

import java.util.Arrays;

public class WaveData {
    private final float[] samples;
    private final int sample_rate;

    public WaveData(float[] samples, int sample_rate) {
        this.samples = samples;
        this.sample_rate = sample_rate;
    }

    public float[] getSamples() {
        return samples;
    }

    public int getSampleRate() {
        return sample_rate;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        WaveData other = (WaveData) obj;
        return sample_rate == other.sample_rate && Arrays.equals(samples, other.samples);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(samples);
        result = 31 * result + sample_rate;
        return result;
    }
}

