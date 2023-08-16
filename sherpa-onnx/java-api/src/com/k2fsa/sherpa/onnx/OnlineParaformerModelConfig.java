/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpa.onnx;

public class OnlineParaformerModelConfig {
  private final String encoder;
  private final String decoder;

  public OnlineParaformerModelConfig(String encoder, String decoder) {
    this.encoder = encoder;
    this.decoder = decoder;
  }

  public String getEncoder() {
    return encoder;
  }

  public String getDecoder() {
    return decoder;
  }
}
