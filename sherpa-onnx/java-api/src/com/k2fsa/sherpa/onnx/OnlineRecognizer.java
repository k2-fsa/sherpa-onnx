// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;


public class OnlineRecognizer {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0; // this is the asr engine ptrss


    public OnlineRecognizer(OnlineRecognizerConfig config) {
        ptr = newFromFile(config);
    }

    public void decode(OnlineStream s) {
        decode(ptr, s.getPtr());
    }


    public boolean isReady(OnlineStream s) {
        return isReady(ptr, s.getPtr());
    }

    public boolean isEndpoint(OnlineStream s) {
        return isEndpoint(ptr, s.getPtr());
    }

    public void reset(OnlineStream s) {
        reset(ptr, s.getPtr());
    }

    public OnlineStream createStream() {
        long p = createStream(ptr, "");
        return new OnlineStream(p);
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

    public OnlineRecognizerResult getResult(OnlineStream s) {
        Object[] arr = getResult(ptr, s.getPtr());
        String text = (String) arr[0];
        String[] tokens = (String[]) arr[1];
        float[] timestamps = (float[]) arr[2];
        return new OnlineRecognizerResult(text, tokens, timestamps);
    }


    private native void delete(long ptr);

    private native long newFromFile(OnlineRecognizerConfig config);

    private native long createStream(long ptr, String hotwords);

    private native void reset(long ptr, long streamPtr);

    private native void decode(long ptr, long streamPtr);

    private native boolean isEndpoint(long ptr, long streamPtr);

    private native boolean isReady(long ptr, long streamPtr);

    private native Object[] getResult(long ptr, long streamPtr);
}