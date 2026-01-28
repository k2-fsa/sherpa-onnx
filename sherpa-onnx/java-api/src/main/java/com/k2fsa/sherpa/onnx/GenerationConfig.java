// Copyright 2026 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

import java.util.Map;

/**
 * Configuration for generating audio.
 * Mirrors Kotlin GenerationConfig.
 */
public class GenerationConfig {

    private float silenceScale = 0.2f;
    private float speed = 1.0f;
    private int sid = 0;

    /** Reference audio samples (mono, [-1, 1]). */
    private float[] referenceAudio = null;

    /** Sample rate of reference audio */
    private int referenceSampleRate = 0;

    /** Optional reference text */
    private String referenceText = null;

    /** Number of steps in flow matching */
    private int numSteps = 5;

    /** Extra model-specific key-value pairs. Can be null. */
    private Map<String, String> extra = null;

    /** Default constructor */
    public GenerationConfig() {
    }

    /** Getters */
    public float getSilenceScale() {
        return silenceScale;
    }

    public float getSpeed() {
        return speed;
    }

    public int getSid() {
        return sid;
    }

    public float[] getReferenceAudio() {
        return referenceAudio;
    }

    public int getReferenceSampleRate() {
        return referenceSampleRate;
    }

    public String getReferenceText() {
        return referenceText;
    }

    public int getNumSteps() {
        return numSteps;
    }

    public Map<String, String> getExtra() {
        return extra;
    }

    /** Setters */
    public void setSilenceScale(float silenceScale) {
        this.silenceScale = silenceScale;
    }

    public void setSpeed(float speed) {
        this.speed = speed;
    }

    public void setSid(int sid) {
        this.sid = sid;
    }

    public void setReferenceAudio(float[] referenceAudio) {
        this.referenceAudio = referenceAudio;
    }

    public void setReferenceSampleRate(int referenceSampleRate) {
        this.referenceSampleRate = referenceSampleRate;
    }

    public void setReferenceText(String referenceText) {
        this.referenceText = referenceText;
    }

    public void setNumSteps(int numSteps) {
        this.numSteps = numSteps;
    }

    public void setExtra(Map<String, String> extra) {
        this.extra = extra;
    }

    @Override
    public String toString() {
        return "GenerationConfig{" +
                "silenceScale=" + silenceScale +
                ", speed=" + speed +
                ", sid=" + sid +
                ", referenceAudioLength=" + (referenceAudio != null ? referenceAudio.length : 0) +
                ", referenceSampleRate=" + referenceSampleRate +
                ", referenceText='" + referenceText + '\'' +
                ", numSteps=" + numSteps +
                ", extra=" + extra +
                '}';
    }
}

