// Copyright 2025 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineSpeechDenoiser {
    private long ptr = 0;

    public OfflineSpeechDenoiser(OfflineSpeechDenoiserConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
    }

    public int getSampleRate() {
        return getSampleRate(ptr);
    }

    public DenoisedAudio run(float[] samples, int sampleRate) {
        return run(ptr, samples, sampleRate);
    }

    protected void finalize() throws Throwable {
        release();
    }

    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    private native void delete(long ptr);

    private native int getSampleRate(long ptr);

    private native DenoisedAudio run(long ptr, float[] samples, int sampleRate);

    private native long newFromFile(OfflineSpeechDenoiserConfig config);
}
