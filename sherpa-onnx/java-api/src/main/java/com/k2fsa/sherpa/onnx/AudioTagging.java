// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class AudioTagging {
    private long ptr = 0;

    public AudioTagging(AudioTaggingConfig config) {
        LibraryLoader.maybeLoad();
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
        return compute(ptr, stream.getPtr(), topK);
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

    private native AudioEvent[] compute(long ptr, long streamPtr, int topK);
}
