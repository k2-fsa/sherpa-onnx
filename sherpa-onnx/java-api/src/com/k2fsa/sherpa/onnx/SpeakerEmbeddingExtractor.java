// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpeakerEmbeddingExtractor {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig config) {
        ptr = newFromFile(config);
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

    public OnlineStream createStream() {
        long p = createStream(ptr);
        return new OnlineStream(p);
    }

    public boolean isReady(OnlineStream s) {
        return isReady(ptr, s.getPtr());
    }

    public float[] compute(OnlineStream s) {
        return compute(ptr, s.getPtr());
    }

    public int getDim() {
        return dim(ptr);
    }

    private native void delete(long ptr);

    private native long newFromFile(SpeakerEmbeddingExtractorConfig config);

    private native long createStream(long ptr);

    private native boolean isReady(long ptr, long streamPtr);

    private native float[] compute(long ptr, long streamPtr);

    private native int dim(long ptr);
}
