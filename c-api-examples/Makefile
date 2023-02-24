
CFLAGS := -I ../
LDFLAGS := -L ../build/lib
LDFLAGS += -L ../build/_deps/onnxruntime-src/lib
LDFLAGS += -lsherpa-onnx-c-api -lsherpa-onnx-core -lonnxruntime -lkaldi-native-fbank-core
LDFLAGS += -Wl,-rpath,../build/lib
LDFLAGS += -Wl,-rpath,../build/_deps/onnxruntime-src/lib

decode-file-c-api: decode-file-c-api.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
