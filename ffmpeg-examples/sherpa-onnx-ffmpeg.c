// ffmpeg-examples/sherpa-onnx-ffmpeg.c
//
// Copyright (c)  2023  Xiaomi Corporation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

/*
 * Copyright (c) 2010 Nicolas George
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2012 Clément Bœsch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file audio decoding and filtering usage example
 * @example sherpa-onnx-ffmpeg.c
 *
 * Demux, decode and filter audio input file, generate a raw audio
 * file to be played with ffplay.
 */

#include <unistd.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
}

static const char *filter_descr =
    "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono";

static AVFormatContext *fmt_ctx;
static AVCodecContext *dec_ctx;
AVFilterContext *buffersink_ctx;
AVFilterContext *buffersrc_ctx;
AVFilterGraph *filter_graph;
static int audio_stream_index = -1;

static int open_input_file(const char *filename) {
  const AVCodec *dec;
  int ret;

  if ((ret = avformat_open_input(&fmt_ctx, filename, NULL, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot open input file %s\n", filename);
    return ret;
  }

  if ((ret = avformat_find_stream_info(fmt_ctx, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
    return ret;
  }

  /* select the audio stream */
  ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR,
           "Cannot find an audio stream in the input file\n");
    return ret;
  }
  audio_stream_index = ret;

  /* create decoding context */
  dec_ctx = avcodec_alloc_context3(dec);
  if (!dec_ctx) return AVERROR(ENOMEM);
  avcodec_parameters_to_context(dec_ctx,
                                fmt_ctx->streams[audio_stream_index]->codecpar);

  /* init the audio decoder */
  if ((ret = avcodec_open2(dec_ctx, dec, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot open audio decoder\n");
    return ret;
  }

  return 0;
}

static int init_filters(const char *filters_descr) {
  char args[512];
  int ret = 0;
  const AVFilter *abuffersrc = avfilter_get_by_name("abuffer");
  const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
  AVFilterInOut *outputs = avfilter_inout_alloc();
  AVFilterInOut *inputs = avfilter_inout_alloc();
  static const enum AVSampleFormat out_sample_fmts[] = {AV_SAMPLE_FMT_S16,
                                                        AV_SAMPLE_FMT_NONE};
  static const int out_sample_rates[] = {16000, -1};
  const AVFilterLink *outlink;
  AVRational time_base = fmt_ctx->streams[audio_stream_index]->time_base;

  filter_graph = avfilter_graph_alloc();
  if (!outputs || !inputs || !filter_graph) {
    ret = AVERROR(ENOMEM);
    goto end;
  }

  /* buffer audio source: the decoded frames from the decoder will be inserted
   * here. */
  if (dec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC)
    av_channel_layout_default(&dec_ctx->ch_layout,
                              dec_ctx->ch_layout.nb_channels);
  ret = snprintf(args, sizeof(args),
                 "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=",
                 time_base.num, time_base.den, dec_ctx->sample_rate,
                 av_get_sample_fmt_name(dec_ctx->sample_fmt));
  av_channel_layout_describe(&dec_ctx->ch_layout, args + ret,
                             sizeof(args) - ret);
  ret = avfilter_graph_create_filter(&buffersrc_ctx, abuffersrc, "in", args,
                                     NULL, filter_graph);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer source\n");
    goto end;
  }

  /* buffer audio sink: to terminate the filter chain. */
  ret = avfilter_graph_create_filter(&buffersink_ctx, abuffersink, "out", NULL,
                                     NULL, filter_graph);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer sink\n");
    goto end;
  }

  ret = av_opt_set_int_list(buffersink_ctx, "sample_fmts", out_sample_fmts, -1,
                            AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output sample format\n");
    goto end;
  }

  ret =
      av_opt_set(buffersink_ctx, "ch_layouts", "mono", AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output channel layout\n");
    goto end;
  }

  ret = av_opt_set_int_list(buffersink_ctx, "sample_rates", out_sample_rates,
                            -1, AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output sample rate\n");
    goto end;
  }

  /*
   * Set the endpoints for the filter graph. The filter_graph will
   * be linked to the graph described by filters_descr.
   */

  /*
   * The buffer source output must be connected to the input pad of
   * the first filter described by filters_descr; since the first
   * filter input label is not specified, it is set to "in" by
   * default.
   */
  outputs->name = av_strdup("in");
  outputs->filter_ctx = buffersrc_ctx;
  outputs->pad_idx = 0;
  outputs->next = NULL;

  /*
   * The buffer sink input must be connected to the output pad of
   * the last filter described by filters_descr; since the last
   * filter output label is not specified, it is set to "out" by
   * default.
   */
  inputs->name = av_strdup("out");
  inputs->filter_ctx = buffersink_ctx;
  inputs->pad_idx = 0;
  inputs->next = NULL;

  if ((ret = avfilter_graph_parse_ptr(filter_graph, filters_descr, &inputs,
                                      &outputs, NULL)) < 0)
    goto end;

  if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0) goto end;

  /* Print summary of the sink buffer
   * Note: args buffer is reused to store channel layout string */
  outlink = buffersink_ctx->inputs[0];
  av_channel_layout_describe(&outlink->ch_layout, args, sizeof(args));
  av_log(NULL, AV_LOG_INFO, "Output: srate:%dHz fmt:%s chlayout:%s\n",
         (int)outlink->sample_rate,
         (char *)av_x_if_null(
             av_get_sample_fmt_name((AVSampleFormat)outlink->format), "?"),
         args);

end:
  avfilter_inout_free(&inputs);
  avfilter_inout_free(&outputs);

  return ret;
}

static void sherpa_decode_frame(const AVFrame *frame,
                                const SherpaOnnxOnlineRecognizer *recognizer,
                                const SherpaOnnxOnlineStream *stream,
                                const SherpaOnnxDisplay *display,
                                int32_t *segment_id) {
#define N 3200  // 100s. Sample rate is fixed to 16 kHz
  static float samples[N];
  static int nb_samples = 0;
  const int16_t *p = (int16_t *)frame->data[0];

  if (frame->nb_samples + nb_samples > N) {
    SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples, nb_samples);
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);
    if (strlen(r->text)) {
      SherpaOnnxPrint(display, *segment_id, r->text);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++*segment_id;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }

    SherpaOnnxDestroyOnlineRecognizerResult(r);
    nb_samples = 0;
  }

  for (int i = 0; i < frame->nb_samples; i++) {
    samples[nb_samples++] = p[i] / 32768.;
  }
}

static inline char *__av_err2str(int errnum) {
  static char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

int main(int argc, char **argv) {
  int ret;
  int num_threads = 1;
  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  AVFrame *filt_frame = av_frame_alloc();
  const char *kUsage =
      "\n"
      "Usage:\n"
      "  ./sherpa-onnx-ffmpeg \\\n"
      "    /path/to/tokens.txt \\\n"
      "    /path/to/encoder.onnx\\\n"
      "    /path/to/decoder.onnx\\\n"
      "    /path/to/joiner.onnx\\\n"
      "    /path/to/foo.wav [num_threads [decoding_method]]"
      "\n\n"
      "Default num_threads is 1.\n"
      "Valid decoding_method: greedy_search (default), modified_beam_search\n\n"
      "Please refer to \n"
      "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html\n"
      "for a list of pre-trained models to download.\n";

  if (!packet || !frame || !filt_frame) {
    fprintf(stderr, "Could not allocate frame or packet\n");
    exit(1);
  }

  if (argc < 6 || argc > 8) {
    fprintf(stderr, "%s\n", kUsage);
    return -1;
  }

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));
  config.model_config.tokens = argv[1];
  config.model_config.transducer.encoder = argv[2];
  config.model_config.transducer.decoder = argv[3];
  config.model_config.transducer.joiner = argv[4];

  if (argc == 7 && atoi(argv[6]) > 0) {
    num_threads = atoi(argv[6]);
  }

  config.model_config.num_threads = num_threads;
  config.model_config.debug = 0;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.decoding_method = "greedy_search";
  if (argc == 8) {
    config.decoding_method = argv[7];
  }

  config.max_active_paths = 4;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&config);
  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxCreateOnlineStream(recognizer);
  const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

  if ((ret = open_input_file(argv[5])) < 0) exit(1);

  if ((ret = init_filters(filter_descr)) < 0) exit(1);

  /* read all packets */
  while (1) {
    if ((ret = av_read_frame(fmt_ctx, packet)) < 0) break;

    if (packet->stream_index == audio_stream_index) {
      ret = avcodec_send_packet(dec_ctx, packet);
      if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR,
               "Error while sending a packet to the decoder\n");
        break;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break;
        } else if (ret < 0) {
          av_log(NULL, AV_LOG_ERROR,
                 "Error while receiving a frame from the decoder\n");
          exit(1);
        }

        if (ret >= 0) {
          /* push the audio data from decoded frame into the filtergraph */
          if (av_buffersrc_add_frame_flags(buffersrc_ctx, frame,
                                           AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
            av_log(NULL, AV_LOG_ERROR,
                   "Error while feeding the audio filtergraph\n");
            break;
          }

          /* pull filtered audio from the filtergraph */
          while (1) {
            ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) exit(1);
            sherpa_decode_frame(filt_frame, recognizer, stream, display,
                                &segment_id);
            av_frame_unref(filt_frame);
          }
          av_frame_unref(frame);
        }
      }
    }
    av_packet_unref(packet);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, tail_paddings, 4800);
  SherpaOnnxOnlineStreamInputFinished(stream);

  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *r =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);
  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);

  avfilter_graph_free(&filter_graph);
  avcodec_free_context(&dec_ctx);
  avformat_close_input(&fmt_ctx);
  av_packet_free(&packet);
  av_frame_free(&frame);
  av_frame_free(&filt_frame);

  if (ret < 0 && ret != AVERROR_EOF) {
    fprintf(stderr, "Error occurred: %s\n", __av_err2str(ret));
    exit(1);
  }
  fprintf(stderr, "\n");

  return 0;
}
