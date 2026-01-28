// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class OnlineRecognizer {
    private long ptr = 0;

    public OnlineRecognizer(OnlineRecognizerConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
    }

    public void decode(OnlineStream s) {
        decode(ptr, s.getPtr());
    }

    public void decode(OnlineStream[] ss) {
        if (ss == null || ss.length == 0) {
          throw new IllegalArgumentException("Stream array must be non-empty");
        }
        long[] streamPtrs = new long[ss.length];
        for (int i = 0; i < ss.length; ++i) {
            streamPtrs[i] = ss[i].getPtr();
        }
        decodeStreams(ptr, streamPtrs);
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
    protected void close()  {
      if (this.ptr == 0) {
        return;
      }
      delete(this.ptr);
      this.ptr = 0;
    }
    
    public void release() {
      this.close();
    }

    public OnlineRecognizerResult getResult(OnlineStream s) {
        return getResult(ptr, s.getPtr());
    }

    private native void delete(long ptr);

    private native long newFromFile(OnlineRecognizerConfig config);

    private native long createStream(long ptr, String hotwords);

    private native void reset(long ptr, long streamPtr);

    private native void decode(long ptr, long streamPtr);

    private native void decodeStreams(long ptr, long[] streamPtrs);

    private native boolean isEndpoint(long ptr, long streamPtr);

    private native boolean isReady(long ptr, long streamPtr);

    private native OnlineRecognizerResult getResult(long ptr, long streamPtr);
}
