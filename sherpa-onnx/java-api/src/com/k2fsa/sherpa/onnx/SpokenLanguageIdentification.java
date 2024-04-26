// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpokenLanguageIdentification {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0; // this is the asr engine ptrss

    // private final localeMap

    public SpokenLanguageIdentification(SpokenLanguageIdentificationConfig config) {
        ptr = newFromFile(config);
    }

    public String compute(OfflineStream stream) {
        return compute(ptr, stream.getPtr());
    }

    public OfflineStream createStream() {
        long p = createStream(ptr);
        return new OfflineStream(p);
    }

    @Override
    protected void finalize() throws Throwable {
        release();
    }

    // You'd better call it manually if it is not used anymore
    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    private native void delete(long ptr);

    private native long newFromFile(SpokenLanguageIdentificationConfig config);

    private native long createStream(long ptr);

    private native String compute(long ptr, long streamPtr);
}
