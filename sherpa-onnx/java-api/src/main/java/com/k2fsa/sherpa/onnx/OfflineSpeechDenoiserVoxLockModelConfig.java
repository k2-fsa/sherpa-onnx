// Copyright 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

/**
 * VoxLock target-speaker voice isolation model config.
 *
 * <p>VoxLock keeps the voice of one speaker (the caller) and removes the other voices. The model is
 * from https://github.com/arbazsiddiqui/voxlock
 */
public class OfflineSpeechDenoiserVoxLockModelConfig {
    private final String enrollModel;
    private final String chunkModel;
    private final String voiceSample;

    private OfflineSpeechDenoiserVoxLockModelConfig(Builder builder) {
        this.enrollModel = builder.enrollModel;
        this.chunkModel = builder.chunkModel;
        this.voiceSample = builder.voiceSample;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getEnrollModel() {
        return enrollModel;
    }

    public String getChunkModel() {
        return chunkModel;
    }

    public String getVoiceSample() {
        return voiceSample;
    }

    public static class Builder {
        private String enrollModel = "";
        private String chunkModel = "";
        private String voiceSample = "";

        public OfflineSpeechDenoiserVoxLockModelConfig build() {
            return new OfflineSpeechDenoiserVoxLockModelConfig(this);
        }

        /**
         * Path to the VoxLock enroll model, e.g., enroll.onnx. It runs once on a voice sample of the
         * speaker to keep.
         */
        public Builder setEnrollModel(String enrollModel) {
            this.enrollModel = enrollModel;
            return this;
        }

        /**
         * Path to the VoxLock chunk model, e.g., chunk.onnx or chunk_int8.onnx. It runs on each chunk
         * of the audio.
         */
        public Builder setChunkModel(String chunkModel) {
            this.chunkModel = chunkModel;
            return this;
        }

        /**
         * Optional. Path to a wave file with at least 3 seconds of the caller's voice. VoxLock keeps
         * this voice and removes the others. From a longer file, it uses the loudest 3 seconds within
         * the first 30 seconds. A file that is shorter or cannot be read is ignored, as if it were
         * empty.
         *
         * <p>If empty or ignored, the voice sample is taken from the input audio: the offline denoiser
         * uses the loudest 3 seconds in the first 15 seconds of the input, and the online denoiser
         * passes the audio through unchanged until 3 seconds of speech have been heard, then uses that
         * speech.
         */
        public Builder setVoiceSample(String voiceSample) {
            this.voiceSample = voiceSample;
            return this;
        }
    }
}
