
ENTRY_POINT = ./

LIB_SRC_DIR := ../sherpa-onnx/java-api/src/com/k2fsa/sherpa/onnx

LIB_FILES = \
	$(LIB_SRC_DIR)/EndpointRule.java \
	$(LIB_SRC_DIR)/EndpointConfig.java \
	$(LIB_SRC_DIR)/FeatureConfig.java \
	$(LIB_SRC_DIR)/OnlineTransducerModelConfig.java \
	$(LIB_SRC_DIR)/OnlineRecognizerConfig.java \
	$(LIB_SRC_DIR)/OnlineStream.java \
	$(LIB_SRC_DIR)/OnlineRecognizer.java \

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
	mkdir -p $(BUILD_DIR)
	mkdir -p ./lib

	
run:

		java -cp ./lib/sherpaonnx.jar:build $(RUNJFLAGS)  DecodeFile 


buildlib: $(LIB_FILES:.java=.class)
	

%.class: %.java

				$(JAVAC) -cp $(BUILD_DIR) -d $(BUILD_DIR) -encoding UTF-8 $<

packjar:
				jar cvfe lib/sherpaonnx.jar . -C $(BUILD_DIR) .
				
all: clean buildlib packjar buildexample run
