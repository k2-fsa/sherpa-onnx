
CUR_DIR :=$(shell pwd)

CFLAGS := -I ../ -I ../build/_deps/cargs-src/include/
LDFLAGS := -L ../build/lib
LDFLAGS += -L ../build/_deps/onnxruntime-src/lib
LDFLAGS += -lsherpa-onnx-c-api -lsherpa-onnx-core -lonnxruntime -lkaldi-native-fbank-core -lkaldi-decoder-core -lsherpa-onnx-kaldifst-core -lsherpa-onnx-fst -lcargs
LDFLAGS += -Wl,-rpath,${CUR_DIR}/../build/lib
LDFLAGS += -Wl,-rpath,${CUR_DIR}/../build/_deps/onnxruntime-src/lib

.PHONY: all clean

all: decode-file-c-api offline-tts-c-api

decode-file-c-api: decode-file-c-api.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

offline-tts-c-api: offline-tts-c-api.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	$(RM) ./decode-file-c-api ./offline-tts-c-api
