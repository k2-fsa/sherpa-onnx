
ENTRY_POINT = ./

LIB_SRC_DIR := ../sherpa-onnx/java-api/src/com/k2fsa/sherpa/onnx

LIB_FILES = \
	$(LIB_SRC_DIR)/EndpointRule.java \
	$(LIB_SRC_DIR)/EndpointConfig.java \
	$(LIB_SRC_DIR)/FeatureConfig.java \
	$(LIB_SRC_DIR)/OnlineLMConfig.java \
	$(LIB_SRC_DIR)/OnlineTransducerModelConfig.java \
	$(LIB_SRC_DIR)/OnlineParaformerModelConfig.java \
	$(LIB_SRC_DIR)/OnlineModelConfig.java \
	$(LIB_SRC_DIR)/OnlineRecognizerConfig.java \
	$(LIB_SRC_DIR)/OnlineStream.java \
	$(LIB_SRC_DIR)/OnlineRecognizer.java \

WEBSOCKET_DIR:= ./src/websocketsrv
WEBSOCKET_FILES = \
	$(WEBSOCKET_DIR)/ConnectionData.java \
	$(WEBSOCKET_DIR)/DecoderThreadHandler.java \
	$(WEBSOCKET_DIR)/StreamThreadHandler.java \
	$(WEBSOCKET_DIR)/AsrWebsocketServer.java \
	$(WEBSOCKET_DIR)/AsrWebsocketClient.java \


LIB_BUILD_DIR = ./lib


EXAMPLE_FILE = DecodeFile.java

EXAMPLE_Mic = DecodeMic.java

JAVAC = javac

BUILD_DIR = build


RUNJFLAGS = -Dfile.encoding=utf-8

vpath %.class $(BUILD_DIR)
vpath %.java src


buildfile:
		$(JAVAC) -cp lib/sherpaonnx.jar -d $(BUILD_DIR) -encoding UTF-8 src/$(EXAMPLE_FILE)

buildmic:
		$(JAVAC) -cp lib/sherpaonnx.jar -d $(BUILD_DIR) -encoding UTF-8 src/$(EXAMPLE_Mic)

rebuild: clean all

.PHONY:  clean run downjar

downjar:
	wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.25/slf4j-api-1.7.25.jar -P ./lib/
	wget https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.25/slf4j-simple-1.7.25.jar -P ./lib/
	wget https://github.com/TooTallNate/Java-WebSocket/releases/download/v1.5.3/Java-WebSocket-1.5.3.jar -P ./lib/


clean:
	rm -frv  $(BUILD_DIR)/*
	rm -frv  $(LIB_BUILD_DIR)/*
	mkdir -p $(BUILD_DIR)
	mkdir -p ./lib

runfile:

		java -cp ./lib/sherpaonnx.jar:build $(RUNJFLAGS)  DecodeFile

runmic:

		java -cp ./lib/sherpaonnx.jar:build $(RUNJFLAGS)  DecodeMic

runsrv:
		java   -cp $(BUILD_DIR):lib/Java-WebSocket-1.5.3.jar:lib/slf4j-simple-1.7.25.jar:lib/slf4j-api-1.7.25.jar:../lib/sherpaonnx.jar  $(RUNJFLAGS)  websocketsrv.AsrWebsocketServer ../build/lib/libsherpa-onnx-jni.so ./modeltest.cfg

runclient:
		java   -cp $(BUILD_DIR):lib/Java-WebSocket-1.5.3.jar:lib/slf4j-simple-1.7.25.jar:lib/slf4j-api-1.7.25.jar:../lib/sherpaonnx.jar  $(RUNJFLAGS)  websocketsrv.AsrWebsocketClient ../build/lib/libsherpa-onnx-jni.so 127.0.0.1 8890 ./test.wav 32

buildlib: $(LIB_FILES:.java=.class)


%.class: %.java

				$(JAVAC) -cp $(BUILD_DIR) -d $(BUILD_DIR) -encoding UTF-8 $<

buildwebsocket: $(WEBSOCKET_FILES:.java=.class)


%.class: %.java

				$(JAVAC) -cp $(BUILD_DIR):lib/slf4j-simple-1.7.25.jar:lib/slf4j-api-1.7.25.jar:lib/Java-WebSocket-1.5.3.jar:../lib/sherpaonnx.jar -d $(BUILD_DIR) -encoding UTF-8 $<

packjar:
				jar cvfe lib/sherpaonnx.jar . -C $(BUILD_DIR) .

all: clean buildlib packjar buildfile buildmic downjar buildwebsocket
