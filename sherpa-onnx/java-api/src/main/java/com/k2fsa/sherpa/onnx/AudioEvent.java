// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class AudioEvent {
    private String name = "";
    private int index = 0;
    private float prob = 0;

    public AudioEvent(String name, int index, float prob) {
        this.name = name;
        this.index = index;
        this.prob = prob;
    }

    public String getName() {
        return name;
    }

    public int getIndex() {
        return index;
    }

    public float getProb() {
        return prob;
    }

    @Override
    public String toString() {
        return String.format("AudioEven(name=%s, index=%d, prob=%.3f)\n", name, index, prob);
    }
}
