// Copyright 2026 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineSpeechDenoiser {
    private long ptr = 0;

    public OnlineSpeechDenoiser(OnlineSpeechDenoiserConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
        if (ptr == 0) {
            throw new IllegalArgumentException("Invalid OnlineSpeechDenoiserConfig: failed to create native OnlineSpeechDenoiser");
        }
    }

    public int getSampleRate() {
        return getSampleRate(ptr);
    }

    public int getFrameShiftInSamples() {
        return getFrameShiftInSamples(ptr);
    }

    public DenoisedAudio run(float[] samples, int sampleRate) {
        return run(ptr, samples, sampleRate);
    }

    public DenoisedAudio flush() {
        return flush(ptr);
    }

    public void reset() {
        reset(ptr);
    }

    @Override
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

    private native int getFrameShiftInSamples(long ptr);

    private native DenoisedAudio run(long ptr, float[] samples, int sampleRate);

    private native DenoisedAudio flush(long ptr);

    private native void reset(long ptr);

    private native long newFromFile(OnlineSpeechDenoiserConfig config);
}
