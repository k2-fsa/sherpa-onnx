# Copyright 2024 Xiaomi Corporation

# all .class and .jar files are put inside out_dir
out_dir := build
out_jar := $(out_dir)/sherpa-onnx.jar

package_dir := com/k2fsa/sherpa/onnx

java_files := LibraryUtils.java
java_files += LibraryLoader.java

java_files += VersionInfo.java
java_files += WaveReader.java
java_files += WaveWriter.java
java_files += EndpointRule.java
java_files += EndpointConfig.java
java_files += FeatureConfig.java
java_files += HomophoneReplacerConfig.java
java_files += OnlineLMConfig.java
java_files += OnlineParaformerModelConfig.java
java_files += OnlineZipformer2CtcModelConfig.java
java_files += OnlineToneCtcModelConfig.java
java_files += OnlineNeMoCtcModelConfig.java
java_files += OnlineTransducerModelConfig.java
java_files += OnlineModelConfig.java
java_files += OnlineCtcFstDecoderConfig.java
java_files += OnlineStream.java
java_files += OnlineRecognizerConfig.java
java_files += OnlineRecognizerResult.java
java_files += OnlineRecognizer.java

java_files += OfflineTransducerModelConfig.java
java_files += OfflineParaformerModelConfig.java
java_files += OfflineWhisperModelConfig.java
java_files += OfflineFireRedAsrModelConfig.java
java_files += OfflineMoonshineModelConfig.java
java_files += OfflineNemoEncDecCtcModelConfig.java
java_files += OfflineZipformerCtcModelConfig.java
java_files += OfflineCanaryModelConfig.java
java_files += OfflineSenseVoiceModelConfig.java
java_files += OfflineDolphinModelConfig.java
java_files += OfflineModelConfig.java
java_files += OfflineRecognizerConfig.java
java_files += OfflineRecognizerResult.java
java_files += OfflineStream.java
java_files += OfflineRecognizer.java

java_files += OfflineTtsKittenModelConfig.java
java_files += OfflineTtsKokoroModelConfig.java
java_files += OfflineTtsMatchaModelConfig.java
java_files += OfflineTtsVitsModelConfig.java
java_files += OfflineTtsModelConfig.java
java_files += OfflineTtsConfig.java
java_files += GeneratedAudio.java
java_files += OfflineTtsCallback.java
java_files += OfflineTts.java

java_files += SpokenLanguageIdentificationWhisperConfig.java
java_files += SpokenLanguageIdentificationConfig.java
java_files += SpokenLanguageIdentification.java

java_files += OfflinePunctuationModelConfig.java
java_files += OfflinePunctuationConfig.java
java_files += OfflinePunctuation.java

java_files += OnlinePunctuationModelConfig.java
java_files += OnlinePunctuationConfig.java
java_files += OnlinePunctuation.java

java_files += OfflineZipformerAudioTaggingModelConfig.java
java_files += AudioTaggingModelConfig.java
java_files += AudioTaggingConfig.java
java_files += AudioEvent.java
java_files += AudioTagging.java

java_files += SpeakerEmbeddingExtractorConfig.java
java_files += SpeakerEmbeddingExtractor.java
java_files += SpeakerEmbeddingManager.java

java_files += TenVadModelConfig.java
java_files += SileroVadModelConfig.java
java_files += VadModelConfig.java
java_files += SpeechSegment.java
java_files += Vad.java

java_files += KeywordSpotterConfig.java
java_files += KeywordSpotterResult.java
java_files += KeywordSpotter.java

java_files += OfflineSpeakerSegmentationPyannoteModelConfig.java
java_files += OfflineSpeakerSegmentationModelConfig.java
java_files += FastClusteringConfig.java
java_files += OfflineSpeakerDiarizationConfig.java
java_files += OfflineSpeakerDiarizationSegment.java
java_files += OfflineSpeakerDiarizationCallback.java
java_files += OfflineSpeakerDiarization.java

java_files += OfflineSpeechDenoiserGtcrnModelConfig.java
java_files += OfflineSpeechDenoiserModelConfig.java
java_files += OfflineSpeechDenoiserConfig.java
java_files += DenoisedAudio.java
java_files += OfflineSpeechDenoiser.java

class_files := $(java_files:%.java=%.class)

java_files := $(addprefix src/main/java/$(package_dir)/,$(java_files))
class_files := $(addprefix $(out_dir)/$(package_dir)/,$(class_files))

$(info -- java files $(java_files))
$(info --)
$(info -- class files $(class_files))

.PHONY: all clean native

all: $(out_jar)

# macos x86_x64 -> osx-x64
# macos arm64 -> osx-aarch64
# linux x86_x64 -> linux-x64
# linux arm64 -> linux-aarch64
# windows x86_x64 -> win-x64
# windows arm64 -> win-aarch64
# windows x86 -> win-x86
native:
	jar cfvm ./sherpa-onnx-native.jar MANIFEST.MF -C ./resources .

$(out_jar): $(class_files)
	# jar --create --verbose --file $(out_jar) -C $(out_dir) ./
	# jar cvf $(out_jar) -C $(out_dir) ./
	jar cfvm $@ MANIFEST.MF -C $(out_dir) .

clean:
	$(RM) -rfv $(out_dir)

$(class_files): $(out_dir)/$(package_dir)/%.class: src/main/java/$(package_dir)/%.java
	mkdir -p build
	javac --release 8 -Xlint:-options -d $(out_dir) -cp $(out_dir) $<
