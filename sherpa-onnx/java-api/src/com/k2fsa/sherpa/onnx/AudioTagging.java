// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class AudioTagging {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr = 0;

    public AudioTagging(AudioTaggingConfig config) {
        ptr = newFromFile(config);
    }

    public OfflineStream createStream() {
        long p = createStream(ptr);
        return new OfflineStream(p);
    }

    public AudioEvent[] compute(OfflineStream stream) {
        return compute(stream, -1);

    }

    public AudioEvent[] compute(OfflineStream stream, int topK) {
        Object[] arr = compute(ptr, stream.getPtr(), topK);

        AudioEvent[] events = new AudioEvent[arr.length];
        for (int i = 0; i < arr.length; ++i) {
            Object[] obj = (Object[]) arr[i];
            String name = (String) obj[0];
            int index = (int) obj[1];
            float prob = (float) obj[2];
            events[i] = new AudioEvent(name, index, prob);
        }
        return events;
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

    private native long newFromFile(AudioTaggingConfig config);

    private native long createStream(long ptr);

    private native Object[] compute(long ptr, long streamPtr, int topK);
}
