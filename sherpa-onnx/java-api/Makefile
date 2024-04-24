
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
java_files += OnlineStream.java
java_files += OnlineRecognizerConfig.java
java_files += OnlineRecognizerResult.java
java_files += OnlineRecognizer.java

class_files := $(java_files:%.java=%.class)

java_files := $(addprefix src/$(package_dir)/,$(java_files))
class_files := $(addprefix $(out_dir)/$(package_dir)/,$(class_files))

$(info -- java files $(java_files))
$(info --)
$(info -- class files $(class_files))

.phony: all clean

all: $(out_jar)

$(out_jar): $(class_files)
	jar --create --verbose --file $(out_jar) -C $(out_dir) .

clean:
	$(RM) -rfv $(out_dir)

$(class_files): $(out_dir)/$(package_dir)/%.class: src/$(package_dir)/%.java
	javac -d $(out_dir) --class-path $(out_dir) $<
