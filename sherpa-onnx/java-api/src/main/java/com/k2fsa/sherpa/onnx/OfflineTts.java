// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

import java.util.function.Consumer;

public class OfflineTts {
    private long ptr = 0;

    public OfflineTts(OfflineTtsConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
        if (ptr == 0) {
            throw new IllegalArgumentException("Invalid OfflineTtsConfig: failed to create native OfflineTts");
        }
    }

    /** Returns the sample rate of the TTS engine. */
    public int getSampleRate() {
        return getSampleRate(ptr);
    }

    public int getNumSpeakers() {
        return getNumSpeakers(ptr);
    }

    /** Generates audio for the given text using the default speaker (sid=0) and speed=1.0. */
    public GeneratedAudio generate(String text) {
        return generate(text, 0, 1.0f);
    }

    /** Generates audio for the given text using a specific speaker ID. */
    public GeneratedAudio generate(String text, int sid) {
        return generate(text, sid, 1.0f);
    }

    /** Generates audio for the given text using a specific speaker ID and speed multiplier. */
    public GeneratedAudio generate(String text, int sid, float speed) {
        return generateImpl(ptr, text, sid, speed);
    }

    public GeneratedAudio generateWithCallback(String text, OfflineTtsCallback callback) {
        return generateWithCallback(text, 0, 1.0f, callback);
    }

    public GeneratedAudio generateWithCallback(
        String text,
        Consumer<float[]> consumer
    ) {
        return generateWithCallback(text, 0, 1.0f, consumer);
    }

    public GeneratedAudio generateWithCallback(String text, int sid, OfflineTtsCallback callback) {
        return generateWithCallback(text, sid, 1.0f, callback);
    }

    public GeneratedAudio generateWithCallback(
        String text,
        int sid,
        Consumer<float[]> consumer
    ) {

        return generateWithCallback(text, sid, 1.0f, consumer);
    }

    public GeneratedAudio generateWithCallback(String text, int sid, float speed, OfflineTtsCallback callback) {
        return generateWithCallbackImpl(ptr, text, sid, speed, callback);
    }

    public GeneratedAudio generateWithCallback(
            String text,
            int sid,
            float speed,
            Consumer<float[]> consumer
    ) {
        OfflineTtsCallback cb = samples -> {
            consumer.accept(samples);
            return 1;
        };
        return generateWithCallback(text, sid, speed, cb);
    }

    /**
     * Generate audio using a GenerationConfig and a callback.
     *
     * @param text The text to synthesize.
     * @param config The generation configuration.
     * @param callback Callback to receive intermediate audio chunks.
     * @return GeneratedAudio with samples and sample rate.
     */
    public GeneratedAudio generateWithConfigAndCallback(
            String text,
            GenerationConfig config,
            OfflineTtsCallback callback
    ) {
        return generateWithConfigImpl(ptr, text, config, callback);
    }


    public GeneratedAudio generateWithConfigAndCallback(
            String text,
            GenerationConfig config,
            Consumer<float[]> consumer
    ) {
        OfflineTtsCallback cb = samples -> {
            consumer.accept(samples);
            return 1;
        };
        return generateWithConfigAndCallback(text, config, cb);
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

    private native int getNumSpeakers(long ptr);

    private native GeneratedAudio generateImpl(long ptr, String text, int sid, float speed);

    private native GeneratedAudio generateWithCallbackImpl(long ptr, String text, int sid, float speed, OfflineTtsCallback callback);

    private native GeneratedAudio generateWithConfigImpl(
            long ptr,
            String text,
            GenerationConfig config,
            OfflineTtsCallback callback
    );

    private native long newFromFile(OfflineTtsConfig config);
}
