/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineTransducerModelConfig {
  private final String encoder;
  private final String decoder;
  private final String joiner;
  private final String tokens;
  private final int numThreads;
  private final boolean debug;
  private final String provider = "cpu";
  private String modelType = "";

  public OnlineTransducerModelConfig(
      String encoder,
      String decoder,
      String joiner,
      String tokens,
      int numThreads,
      boolean debug,
      String modelType) {
    this.encoder = encoder;
    this.decoder = decoder;
    this.joiner = joiner;
    this.tokens = tokens;
    this.numThreads = numThreads;
    this.debug = debug;
    this.modelType = modelType;
  }

  public String getEncoder() {
    return encoder;
  }

  public String getDecoder() {
    return decoder;
  }

  public String getJoiner() {
    return joiner;
  }

  public String getTokens() {
    return tokens;
  }

  public int getNumThreads() {
    return numThreads;
  }

  public boolean getDebug() {
    return debug;
  }
}
