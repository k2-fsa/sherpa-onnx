CC=g++
GDB ?= FALSE

# use pkg-config for getting CFLAGS and LDLIBS
SHARED_LIBS=libavdevice                          \
            libavformat                          \
            libavfilter                          \
            libavcodec                           \
            libswresample                        \
            libswscale                           \
            libavutil

ifeq ($(GDB), TRUE)
	OPTFLAG += -g
endif

# CFLAGS := $(shell pkg-config --cflags $(SHARED_LIBS)) -I.. -Wall -std=c++17 -fopenmp ${OPTFLAG}
CFLAGS := $(shell pkg-config --cflags $(SHARED_LIBS)) -I.. -Wall -std=c++17  ${OPTFLAG}
LDLIBS := $(shell pkg-config --libs $(SHARED_LIBS))

CUR_DIR :=$(shell pwd)

LDLIBS += -L ../build/lib
LDLIBS += -L ../build/_deps/onnxruntime-src/lib
LDLIBS += -lsherpa-onnx-c-api -lonnxruntime
LDLIBS += -Wl,-rpath,${CUR_DIR}/../build/lib
LDLIBS += -Wl,-rpath,${CUR_DIR}/../build/_deps/onnxruntime-src/lib

#Get libavutil version and extract major, minor and micro
LIBAVUTIL_VERSION := $(shell pkg-config --modversion libavutil)
LIBAVUTIL_MAJOR := $(shell echo "$(LIBAVUTIL_VERSION)" | awk -F. '{print $$1}')
LIBAVUTIL_MINOR := $(shell echo "$(LIBAVUTIL_VERSION)" | awk -F. '{print $$2}')
LIBAVUTIL_MICRO := $(shell echo "$(LIBAVUTIL_VERSION)" | awk -F. '{print $$3}')
#Check if libavutil version is 57.28.100 or above
FFMPEG_51_AND_ABOVE = $(shell echo "$(LIBAVUTIL_MAJOR) $(LIBAVUTIL_MINOR) $(LIBAVUTIL_MICRO)" | awk '{if ($$1 > 57 || ($$1 == 57 && $$2 > 28) || ($$1 == 57 && $$2 == 28 && $$3 >= 100)) print "TRUE"; else print "FALSE"}')
ifeq ($(FFMPEG_51_AND_ABOVE), FALSE)
$(error FFmpeg version should be n5.1 or above!)
endif

EXAMPLES=sherpa-onnx-ffmpeg

OBJS=$(addsuffix .o,$(EXAMPLES))

.phony: all clean

all: $(EXAMPLES)
	@echo $(EXAMPLES)
	$(RM) $(OBJS)

$(EXAMPLES): $(OBJS)
	$(CC) $(addsuffix .o,$@) $(CFLAGS) $(LDLIBS) -o $@

%.o : %.c
	${CC} ${CFLAGS} -c -o $@ $<

clean:
	$(RM) $(EXAMPLES) $(OBJS)

build_info:
	@echo "libavutil version: $(LIBAVUTIL_VERSION)"
	@echo "Supported examples: $(EXAMPLES)"
