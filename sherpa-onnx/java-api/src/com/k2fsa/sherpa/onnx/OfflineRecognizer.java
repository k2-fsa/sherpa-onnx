// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OfflineRecognizer {
    private long ptr = 0;

    public OfflineRecognizer(OfflineRecognizerConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
    }

    public void decode(OfflineStream s) {
        decode(ptr, s.getPtr());
    }

    public void decode(OfflineStream[] ss) {
        long[] streamPtrs = new long[ss.length];
        for (int i = 0; i < ss.length; ++i) {
            streamPtrs[i] = ss[i].getPtr();
        }
        decodeStreams(ptr, streamPtrs);
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
        String lang = (String) arr[3];
        String emotion = (String) arr[4];
        String event = (String) arr[5];
        return new OfflineRecognizerResult(text, tokens, timestamps, lang, emotion, event);
    }

    private native void delete(long ptr);

    private native long newFromFile(OfflineRecognizerConfig config);

    private native long createStream(long ptr);

    private native void decode(long ptr, long streamPtr);

    private native void decodeStreams(long ptr, long[] streamPtrs);

    private native Object[] getResult(long streamPtr);
}
