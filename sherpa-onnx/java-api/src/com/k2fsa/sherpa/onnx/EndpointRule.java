/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class EndpointRule {
  private final boolean mustContainNonSilence;
  private final float minTrailingSilence;
  private final float minUtteranceLength;

  public EndpointRule(
      boolean mustContainNonSilence, float minTrailingSilence, float minUtteranceLength) {
    this.mustContainNonSilence = mustContainNonSilence;
    this.minTrailingSilence = minTrailingSilence;
    this.minUtteranceLength = minUtteranceLength;
  }

  public float getMinTrailingSilence() {
    return minTrailingSilence;
  }

  public float getMinUtteranceLength() {
    return minUtteranceLength;
  }

  public boolean getMustContainNonSilence() {
    return mustContainNonSilence;
  }
}
