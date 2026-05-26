// Copyright 2025 Xiaomi Corporation

// This file shows how to use DPDFNet speech enhancement models in sherpa-onnx
//
// Download DPDFNet models from either:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// https://huggingface.co/Ceva-IP/DPDFNet
//
// Use dpdfnet_baseline.onnx, dpdfnet2.onnx, dpdfnet4.onnx, or dpdfnet8.onnx
// for 16 kHz downstream ASR or speech recognition.
// Use dpdfnet2_48khz_hr.onnx for 48 kHz enhancement output.

import com.k2fsa.sherpa.onnx.*;

public class NonStreamingSpeechEnhancementDpdfNet {
  public static void main(String[] args) {
    String model = "./dpdfnet_baseline.onnx";
    OfflineSpeechDenoiserModelConfig.Builder builder =
        OfflineSpeechDenoiserModelConfig.builder()
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .setDpdfnet(
                OfflineSpeechDenoiserDpdfNetModelConfig.builder().setModel(model).build());

    OfflineSpeechDenoiserModelConfig modelConfig = builder.build();
    OfflineSpeechDenoiserConfig config =
        OfflineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OfflineSpeechDenoiser speech_denoiser = new OfflineSpeechDenoiser(config);

    String testWaveFilename = "./inp_16k.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    DenoisedAudio denoised = speech_denoiser.run(reader.getSamples(), reader.getSampleRate());
    String outFilename = "enhanced.wav";
    WaveWriter.write(outFilename, denoised.getSamples(), denoised.getSampleRate());
    System.out.printf("Saved to %s\n", outFilename);

    speech_denoiser.release();
  }
}
