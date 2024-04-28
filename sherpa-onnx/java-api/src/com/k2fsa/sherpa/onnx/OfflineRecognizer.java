// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineRecognizer {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0; // this is the asr engine ptrss

    public OfflineRecognizer(OfflineRecognizerConfig config) {
        ptr = newFromFile(config);
    }

    public void decode(OfflineStream s) {
        decode(ptr, s.getPtr());
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

    public OfflineRecognizerResult getResult(OfflineStream s) {
        Object[] arr = getResult(s.getPtr());
        String text = (String) arr[0];
        String[] tokens = (String[]) arr[1];
        float[] timestamps = (float[]) arr[2];
        return new OfflineRecognizerResult(text, tokens, timestamps);
    }

    private native void delete(long ptr);

    private native long newFromFile(OfflineRecognizerConfig config);

    private native long createStream(long ptr);

    private native void decode(long ptr, long streamPtr);

    private native Object[] getResult(long streamPtr);
}
