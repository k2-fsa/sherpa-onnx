// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class SpokenLanguageIdentification {
    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private final Map<String, String> localeMap;
    private long ptr = 0;

    public SpokenLanguageIdentification(SpokenLanguageIdentificationConfig config) {
        ptr = newFromFile(config);

        String[] languages = Locale.getISOLanguages();
        localeMap = new HashMap<String, String>(languages.length);
        for (String language : languages) {
            Locale locale = new Locale(language);
            localeMap.put(language, locale.getDisplayName());
        }
    }

    public String compute(OfflineStream stream) {
        String lang = compute(ptr, stream.getPtr());
        return localeMap.getOrDefault(lang, lang);
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
