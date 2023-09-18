
CUR_DIR :=$(shell pwd)

CFLAGS := -I ../ -I ../build/_deps/cargs-src/include/
LDFLAGS := -L ../build/lib
LDFLAGS += -L ../build/_deps/onnxruntime-src/lib
LDFLAGS += -lsherpa-onnx-c-api -lsherpa-onnx-core -lonnxruntime -lkaldi-native-fbank-core -lcargs
LDFLAGS += -Wl,-rpath,${CUR_DIR}/../build/lib
LDFLAGS += -Wl,-rpath,${CUR_DIR}/../build/_deps/onnxruntime-src/lib

decode-file-c-api: decode-file-c-api.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
