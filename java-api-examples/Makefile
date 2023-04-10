
ENTRY_POINT = ./

LIB_FILES = \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/FeatureConfig.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/EndpointRule.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/EndpointConfig.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/OnlineTransducerModelConfig.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/EndpointRule.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/OnlineTransducerModelConfig.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/OnlineRecognizerConfig.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/OnlineStream.java \
	../sherpa-onnx/java-api/src/com/k2fsa/sherpaonnx/OnlineRecognizer.java \

LIB_BUILD_DIR = ./lib


EXAMPLE_FILE = DecodeFile.java

JAVAC = javac

BUILD_DIR = build


RUNJFLAGS = -Dfile.encoding=utf-8
	
	
vpath %.class $(BUILD_DIR)
vpath %.java src
	
	
buildexample:  
		$(JAVAC) -cp lib/sherpaonnx.jar -d $(BUILD_DIR) -encoding UTF-8 src/$(EXAMPLE_FILE)
	
rebuild: clean all
	
.PHONY:  clean run
	
clean:
	rm -frv  $(BUILD_DIR)/*
	rm -frv  $(LIB_BUILD_DIR)/*

	
run:
		java -cp ./lib/sherpaonnx.jar:build $(RUNJFLAGS)  DecodeFile 


buildlib: $(LIB_FILES:.java=.class)
	

%.class: %.java
				$(JAVAC) -cp $(BUILD_DIR) -d $(BUILD_DIR) -encoding UTF-8 $<

packjar:
				jar cvfe lib/sherpaonnx.jar . -C $(BUILD_DIR) .
				
all: clean buildlib packjar buildexample run