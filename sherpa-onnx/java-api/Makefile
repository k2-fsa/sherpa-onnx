
# all .class and .jar files are put inside out_dir
out_dir := build
out_jar := $(out_dir)/sherpa-onnx.jar

package_dir := com/k2fsa/sherpa/onnx

java_files := WaveReader.java
java_files += EndpointRule.java
java_files += EndpointConfig.java
java_files += FeatureConfig.java
java_files += OnlineLMConfig.java
java_files += OnlineParaformerModelConfig.java
java_files += OnlineZipformer2CtcModelConfig.java
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
java_files += OfflineNemoEncDecCtcModelConfig.java
java_files += OfflineModelConfig.java
java_files += OfflineRecognizerConfig.java
java_files += OfflineRecognizerResult.java
java_files += OfflineStream.java
java_files += OfflineRecognizer.java

class_files := $(java_files:%.java=%.class)

java_files := $(addprefix src/$(package_dir)/,$(java_files))
class_files := $(addprefix $(out_dir)/$(package_dir)/,$(class_files))

$(info -- java files $(java_files))
$(info --)
$(info -- class files $(class_files))

.phony: all clean

all: $(out_jar)

$(out_jar): $(class_files)
	# jar --create --verbose --file $(out_jar) -C $(out_dir) ./
	jar cvf $(out_jar) -C $(out_dir) ./

clean:
	$(RM) -rfv $(out_dir)

$(class_files): $(out_dir)/$(package_dir)/%.class: src/$(package_dir)/%.java
	mkdir -p build
	javac -d $(out_dir) -cp $(out_dir) $<
