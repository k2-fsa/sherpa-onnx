/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineModelConfig {
  private final OnlineParaformerModelConfig paraformer;
  private final OnlineTransducerModelConfig transducer;
  private final String tokens;
  private final int numThreads;
  private final boolean debug;
  private final String provider = "cpu";
  private String modelType = "";

  public OnlineModelConfig(
      String tokens,
      int numThreads,
      boolean debug,
      String modelType,
      OnlineParaformerModelConfig paraformer,
      OnlineTransducerModelConfig transducer) {

    this.tokens = tokens;
    this.numThreads = numThreads;
    this.debug = debug;
    this.modelType = modelType;
    this.paraformer = paraformer;
    this.transducer = transducer;
  }

  public OnlineParaformerModelConfig getParaformer() {
    return paraformer;
  }

  public OnlineTransducerModelConfig getTransducer() {
    return transducer;
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
