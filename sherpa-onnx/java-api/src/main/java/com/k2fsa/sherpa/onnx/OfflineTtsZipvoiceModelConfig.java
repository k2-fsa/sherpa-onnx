package com.k2fsa.sherpa.onnx;

public class OfflineTtsZipvoiceModelConfig {
  private final String tokens;
  private final String textModel;
  private final String flowMatchingModel;
  private final String vocoder;
  private final String dataDir;
  private final String pinyinDict;
  private final float featScale;
  private final float tShift;
  private final float targetRms;
  private final float guidanceScale;

  private OfflineTtsZipvoiceModelConfig(Builder builder) {
    this.tokens = builder.tokens;
    this.textModel = builder.textModel;
    this.flowMatchingModel = builder.flowMatchingModel;
    this.vocoder = builder.vocoder;
    this.dataDir = builder.dataDir;
    this.pinyinDict = builder.pinyinDict;
    this.featScale = builder.featScale;
    this.tShift = builder.tShift;
    this.targetRms = builder.targetRms;
    this.guidanceScale = builder.guidanceScale;

  }

  public static Builder builder() {
    return new Builder();
  }

  public String getTokens() {
    return tokens;
  }

  public String getTextModel() {
    return textModel;
  }

  public String getFlowMatchingModel() {
    return flowMatchingModel;
  }

  public String getVocoder() {
    return vocoder;
  }

  public String getDataDir() {
    return dataDir;
  }

  public String getPinyinDict() {
    return pinyinDict;
  }

  public float getFeatScale() {
    return featScale;
  }

  public float getTShift() {
    return tShift;
  }

  public float getTargetRms() {
    return targetRms;
  }

  public float getGuidanceScale() {
    return guidanceScale;
  }

  public static class Builder {
    private String tokens = "";
    private String textModel = "";
    private String flowMatchingModel = "";
    private String vocoder = "";
    private String dataDir = "";
    private String pinyinDict = "";
    private float featScale = 0.1f;
    private float tShift = 0.5f;
    private float targetRms = 0.1f;
    private float guidanceScale = 1.0f;

    public OfflineTtsZipvoiceModelConfig build() {
      return new OfflineTtsZipvoiceModelConfig(this);
    }

    public Builder setTokens(String tokens) {
      this.tokens = tokens;
      return this;
    }

    public Builder setTextModel(String textModel) {
      this.textModel = textModel;
      return this;
    }

    public Builder setFlowMatchingModel(String flowMatchingModel) {
      this.flowMatchingModel = flowMatchingModel;
      return this;
    }

    public Builder setVocoder(String vocoder) {
      this.vocoder = vocoder;
      return this;
    }

    public Builder setDataDir(String dataDir) {
      this.dataDir = dataDir;
      return this;
    }

    public Builder setPinyinDict(String pinyinDict) {
      this.pinyinDict = pinyinDict;
      return this;
    }

    public Builder setFeatScale(float featScale) {
      this.featScale = featScale;
      return this;
    }

    public Builder setTShift(float tShift) {
      this.tShift = tShift;
      return this;
    }

    public Builder setTargetRms(float targetRms) {
      this.targetRms = targetRms;
      return this;
    }

    public Builder setGuidanceScale(float guidanceScale) {
      this.guidanceScale = guidanceScale;
      return this;
    }

  }
}
