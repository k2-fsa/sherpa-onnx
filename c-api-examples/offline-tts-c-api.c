// c-api-examples/offline-tts-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to convert text to speech using an offline model.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cargs.h"
#include "portaudio.h"
#include "sherpa-onnx/c-api/c-api.h"

static struct cag_option options[] = {
    {.identifier = 'h',
     .access_letters = "h",
     .access_name = "help",
     .description = "Show help"},
    {.access_name = "vits-model",
     .value_name = "/path/to/xxx.onnx",
     .identifier = '0',
     .description = "Path to VITS model"},
    {.access_name = "vits-lexicon",
     .value_name = "/path/to/lexicon.txt",
     .identifier = '1',
     .description = "Path to lexicon.txt for VITS models"},
    {.access_name = "vits-tokens",
     .value_name = "/path/to/tokens.txt",
     .identifier = '2',
     .description = "Path to tokens.txt for VITS models"},
    {.access_name = "vits-noise-scale",
     .value_name = "0.667",
     .identifier = '3',
     .description = "noise_scale for VITS models"},
    {.access_name = "vits-noise-scale-w",
     .value_name = "0.8",
     .identifier = '4',
     .description = "noise_scale_w for VITS models"},
    {.access_name = "vits-length-scale",
     .value_name = "1.0",
     .identifier = '5',
     .description =
         "length_scale for VITS models. Default to 1. You can tune it "
         "to change the speech speed. small -> faster; large -> slower. "},
    {.access_name = "num-threads",
     .value_name = "1",
     .identifier = '6',
     .description = "Number of threads"},
    {.access_name = "provider",
     .value_name = "cpu",
     .identifier = '7',
     .description = "Provider: cpu (default), cuda, coreml"},
    {.access_name = "debug",
     .value_name = "0",
     .identifier = '8',
     .description = "1 to show debug messages while loading the model"},
    {.access_name = "sid",
     .value_name = "0",
     .identifier = '9',
     .description = "Speaker ID. Default to 0. Note it is not used for "
                    "single-speaker models."},
    {.access_name = "output-filename",
     .value_name = "./generated.wav",
     .identifier = 'a',
     .description =
         "Filename to save the generated audio. Default to ./generated.wav"},
};

static void ShowUsage() {
  const char *kUsageMessage =
      "Offline text-to-speech with sherpa-onnx C API"
      "\n"
      "./offline-tts-c-api \\\n"
      " --vits-model=/path/to/model.onnx \\\n"
      " --vits-lexicon=/path/to/lexicon.txt \\\n"
      " --vits-tokens=/path/to/tokens.txt \\\n"
      " --sid=0 \\\n"
      " --output-filename=./generated.wav \\\n"
      " 'some text within single quotes on linux/macos or use double quotes on "
      "windows'\n"
      "\n"
      "It will generate a file ./generated.wav as specified by "
      "--output-filename.\n"
      "\n"
      "You can download a test model from\n"
      "https://huggingface.co/csukuangfj/vits-ljs\n"
      "\n"
      "For instance, you can use:\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt\n"
      "wget "
      "https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt\n"
      "\n"
      "./offline-tts-c-api \\\n"
      "  --vits-model=./vits-ljs.onnx \\\n"
      "  --vits-lexicon=./lexicon.txt \\\n"
      "  --vits-tokens=./tokens.txt \\\n"
      "  --sid=0 \\\n"
      "  --output-filename=./generated.wav \\\n"
      "  'liliana, the most beautiful and lovely assistant of our team!'\n"
      "\n"
      "Please see\n"
      "https://k2-fsa.github.io/sherpa/onnx/tts/index.html\n"
      "or details.\n\n";

  fprintf(stderr, "%s", kUsageMessage);
  cag_option_print(options, CAG_ARRAY_SIZE(options), stderr);
  exit(0);
}

int32_t g_i = 0;
static int patestCallback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags, void *userData) {
  SherpaOnnxGeneratedAudio *data = (SherpaOnnxGeneratedAudio *)userData;
  float *out = (float *)outputBuffer;

  (void)timeInfo; /* Prevent unused variable warnings. */
  (void)statusFlags;
  (void)inputBuffer;

  for (int32_t i = 0; i < framesPerBuffer; i++) {
    if (g_i >= data->n) {
      return paComplete;
    }

    *out++ = data->samples[g_i + i]; /* left */
  }
  g_i += framesPerBuffer;

  return paContinue;
}

static void StreamFinished(void *userData) {
  SherpaOnnxGeneratedAudio *data = (SherpaOnnxGeneratedAudio *)userData;
  printf("Stream Completed. Number of samples %d\n", data->n);
}

#define FRAMES_PER_BUFFER (64)

PaError Play(const SherpaOnnxGeneratedAudio *audio) {
  printf("n: %d, sample_rate: %d\n", audio->n, audio->sample_rate);
  PaStreamParameters outputParameters;
  PaStream *stream;
  PaError err;

  err = Pa_Initialize();
  if (err != paNoError) goto error;

  outputParameters.device =
      Pa_GetDefaultOutputDevice(); /* default output device */

  if (outputParameters.device == paNoDevice) {
    fprintf(stderr, "Error: No default output device.\n");
    goto error;
  }

  const PaDeviceInfo *info = Pa_GetDeviceInfo(outputParameters.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max output channels: %d\n", info->maxOutputChannels);

  outputParameters.channelCount = 1;         /* Mono output */
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency =
      Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  err = Pa_OpenStream(&stream, NULL, /* no input */
                      &outputParameters, audio->sample_rate, FRAMES_PER_BUFFER,
                      paClipOff, /* we won't output out of range samples so
                                    don't bother clipping them */
                      patestCallback, (void *)audio);
  if (err != paNoError) goto error;

  err = Pa_SetStreamFinishedCallback(stream, &StreamFinished);
  if (err != paNoError) goto error;

  err = Pa_StartStream(stream);
  if (err != paNoError) goto error;

  float num_seconds = audio->n / (float)audio->sample_rate;
  printf("Play for %.3f seconds.\n", num_seconds);
  while (Pa_IsStreamActive(stream) == 1) {
    Pa_Sleep(0.1 * 1000);  // 100ms
  }

  err = Pa_StopStream(stream);
  if (err != paNoError) goto error;
  err = Pa_CloseStream(stream);
  if (err != paNoError) goto error;

  Pa_Terminate();
  printf("Test finished.\n");

  return err;
error:
  Pa_Terminate();
  fprintf(stderr, "An error occurred while using the portaudio stream\n");
  fprintf(stderr, "Error number: %d\n", err);
  fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
  return err;
}

int32_t main(int32_t argc, char *argv[]) {
  cag_option_context context;
  char identifier;
  const char *value;

  cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);

  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));

  int32_t sid = 0;
  const char *filename = strdup("./generated.wav");
  const char *text;

  while (cag_option_fetch(&context)) {
    identifier = cag_option_get(&context);
    value = cag_option_get_value(&context);
    switch (identifier) {
      case '0':
        config.model.vits.model = value;
        break;
      case '1':
        config.model.vits.lexicon = value;
        break;
      case '2':
        config.model.vits.tokens = value;
        break;
      case '3':
        config.model.vits.noise_scale = atof(value);
        break;
      case '4':
        config.model.vits.noise_scale_w = atof(value);
        break;
      case '5':
        config.model.vits.length_scale = atof(value);
        break;
      case '6':
        config.model.num_threads = atoi(value);
        break;
      case '7':
        config.model.provider = value;
        break;
      case '8':
        config.model.debug = atoi(value);
        break;
      case '9':
        sid = atoi(value);
        break;
      case 'a':
        free((void *)filename);
        filename = strdup(value);
        break;
      case 'h':
        // fall through
      default:
        ShowUsage();
    }
  }

  if (!config.model.vits.model || !config.model.vits.lexicon ||
      !config.model.vits.tokens) {
    ShowUsage();
  }

  // the last arg is the text
  text = argv[argc - 1];
  if (text[0] == '-') {
    fprintf(stderr, "\n***Please input your text!***\n\n");
    fprintf(stderr, "\n---------------Usage---------------\n\n");
    ShowUsage();
  }

  SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);

  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerate(tts, text, sid, 1.0);

  SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);

  Play(audio);
  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  SherpaOnnxDestroyOfflineTts(tts);

  fprintf(stderr, "Input text is: %s\n", text);
  fprintf(stderr, "Speaker ID is is: %d\n", sid);
  fprintf(stderr, "Saved to: %s\n", filename);

  free((void *)filename);

  return 0;
}
