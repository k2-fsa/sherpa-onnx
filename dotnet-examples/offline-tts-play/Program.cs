// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a non-streaming TTS model for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
//
// Note that you need a speaker to run this file since it will play
// the generated audio as it is generating.

using CommandLine;
using CommandLine.Text;
using PortAudioSharp;
using SherpaOnnx;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

class OfflineTtsPlayDemo
{
  class Options
  {
    [Option("tts-rule-fsts", Required = false, Default = "", HelpText = "path to rule.fst")]
    public string RuleFsts { get; set; } = string.Empty;

    [Option("tts-rule-fars", Required = false, Default = "", HelpText = "path to rule.far")]
    public string RuleFars { get; set; } = string.Empty;

    [Option("data-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for espeak-ng.")]
    public string DataDir { get; set; } = string.Empty;

    [Option("length-scale", Required = false, Default = 1, HelpText = "speech speed. Larger->Slower; Smaller->faster")]
    public float LengthScale { get; set; } = 1;

    [Option("noise-scale", Required = false, Default = 0.667f, HelpText = "noise_scale for VITS or Matcha models")]
    public float NoiseScale { get; set; } = 0.667F;

    [Option("vits-noise-scale-w", Required = false, Default = 0.8F, HelpText = "noise_scale_w for VITS models")]
    public float NoiseScaleW { get; set; } = 0.8F;

    [Option("lexicon", Required = false, Default = "", HelpText = "Path to lexicon.txt")]
    public string Lexicon { get; set; } = string.Empty;

    [Option("tokens", Required = true, Default = "", HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; } = string.Empty;

    [Option("tts-max-num-sentences", Required = false, Default = 1, HelpText = "Maximum number of sentences that we process at a time.")]
    public int MaxNumSentences { get; set; } = 1;

    [Option(Required = false, Default = 0, HelpText = "1 to show debug messages.")]
    public int Debug { get; set; } = 0;

    [Option("vits-model", Required = false, HelpText = "Path to VITS model")]
    public string Model { get; set; } = string.Empty;

    [Option("matcha-acoustic-model", Required = false, HelpText = "Path to the acoustic model of Matcha")]
    public string AcousticModel { get; set; } = "";

    [Option("matcha-vocoder", Required = false, HelpText = "Path to the vocoder model of Matcha")]
    public string Vocoder { get; set; } = "";

    [Option("zipvoice-text-model", Required = false, Default = "", HelpText = "Path to ZipVoice text encoder model")]
    public string ZipvoiceTextModel { get; set; } = string.Empty;

    [Option("zipvoice-flow-matching-model", Required = false, Default = "", HelpText = "Path to ZipVoice flow-matching decoder")]
    public string ZipvoiceFlowMatchingModel { get; set; } = string.Empty;

    [Option("zipvoice-data-dir", Required = false, Default = "", HelpText = "Path to espeak-ng-data for ZipVoice")]
    public string ZipvoiceDataDir { get; set; } = string.Empty;

    [Option("zipvoice-pinyin-dict", Required = false, Default = "", HelpText = "Path to pinyin.raw (for zh)")]
    public string ZipvoicePinyinDict { get; set; } = string.Empty;

    [Option("zipvoice-vocoder", Required = false, Default = "", HelpText = "Path to vocoder for ZipVoice (e.g., vocos_24khz.onnx)")]
    public string ZipvoiceVocoder { get; set; } = string.Empty;

    [Option("zipvoice-feat-scale", Required = false, Default = 0.1f, HelpText = "Feature scale for ZipVoice")]
    public float ZipvoiceFeatScale { get; set; } = 0.1f;

    [Option("zipvoice-t-shift", Required = false, Default = 0.5f, HelpText = "t-shift for ZipVoice (smaller -> earlier t)")]
    public float ZipvoiceTShift { get; set; } = 0.5f;

    [Option("zipvoice-target-rms", Required = false, Default = 0.1f, HelpText = "Target RMS for speech normalization (ZipVoice)")]
    public float ZipvoiceTargetRms { get; set; } = 0.1f;

    [Option("zipvoice-guidance-scale", Required = false, Default = 1.0f, HelpText = "Classifier-free guidance scale (ZipVoice)")]
    public float ZipvoiceGuidanceScale { get; set; } = 1.0f;

    [Option("zipvoice-num-steps", Required = false, Default = 4, HelpText = "Number of steps for ZipVoice inference")]
    public int ZipvoiceNumSteps { get; set; } = 4;

    [Option("prompt-text", Required = false, Default = "", HelpText = "Transcription of the prompt audio (ZipVoice)")]
    public string PromptText { get; set; } = string.Empty;

    [Option("prompt-audio", Required = false, Default = "", HelpText = "Path to prompt audio wav (ZipVoice)")]
    public string PromptAudio { get; set; } = string.Empty;

    [Option("sid", Required = false, Default = 0, HelpText = "Speaker ID")]
    public int SpeakerId { get; set; } = 0;

    [Option("text", Required = true, HelpText = "Text to synthesize")]
    public string Text { get; set; } = string.Empty;

    [Option("output-filename", Required = true, Default = "./generated.wav", HelpText = "Path to save the generated audio")]
    public string OutputFilename { get; set; } = "./generated.wav";
  }

  static void Main(string[] args)
  {
    var parser = new CommandLine.Parser(with => with.HelpWriter = null);
    var parserResult = parser.ParseArguments<Options>(args);

    parserResult
      .WithParsed<Options>(options => Run(options))
      .WithNotParsed(errs => DisplayHelp(parserResult, errs));
  }

  private static void DisplayHelp<T>(ParserResult<T> result, IEnumerable<Error> errs)
  {
    string usage = @"
# matcha-icefall-zh-baker

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

dotnet run \
  --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
  --matcha-vocoder=./vocos-22khz-univ.onnx \
  --lexicon=./matcha-icefall-zh-baker/lexicon.txt \
  --tokens=./matcha-icefall-zh-baker/tokens.txt \
  --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
  --debug=1 \
  --output-filename=./matcha-zh.wav \
  --text='某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。'

# matcha-icefall-en_US-ljspeech

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

dotnet run \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./vocos-22khz-univ.onnx \
  --tokens=./matcha-icefall-zh-baker/tokens.txt \
  --data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
  --debug=1 \
  --output-filename=./matcha-zh.wav \
  --text='Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'

# vits-aishell3

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
tar xf vits-zh-aishell3.tar.bz2

dotnet run \
  --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
  --tokens=./vits-zh-aishell3/tokens.txt \
  --lexicon=./vits-zh-aishell3/lexicon.txt \
  --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
  --sid=66 \
  --debug=1 \
  --output-filename=./aishell3-66.wav \
  --text=这是一个语音合成测试

# Piper models

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

dotnet run \
  --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  ---tokens=./vits-piper-en_US-amy-low/tokens.txt \
  --data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
  --debug=1 \
  --output-filename=./amy.wav \
  --text='This is a text to speech application in dotnet with Next Generation Kaldi'

Please refer to
https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
to download more models.

# ZipVoice (zh-en emilia)

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2

dotnet run \
  --zipvoice-flow-matching-model=./sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx \
  --zipvoice-text-model=./sherpa-onnx-zipvoice-distill-zh-en-emilia/text_encoder.onnx \
  --tokens=./sherpa-onnx-zipvoice-distill-zh-en-emilia/tokens.txt \
  --zipvoice-data-dir=./sherpa-onnx-zipvoice-distill-zh-en-emilia/espeak-ng-data \
  --zipvoice-pinyin-dict=./sherpa-onnx-zipvoice-distill-zh-en-emilia/pinyin.raw \
  --zipvoice-vocoder=./sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx \
  --prompt-audio=./sherpa-onnx-zipvoice-distill-zh-en-emilia/prompt.wav \
  --prompt-text=周日被我射熄火了，所以今天是周一。 \
  --zipvoice-num-steps=4 \
  --debug=1 \
  --output-filename=./test-zipvoice.wav \
  --text=我是中国人民的儿子，我爱我的祖国。我的祖国是一个伟大的国家，拥有五千年的文明史。
";

    var helpText = HelpText.AutoBuild(result, h =>
    {
      h.AdditionalNewLineAfterOption = false;
      h.Heading = usage;
      h.Copyright = "Copyright (c) 2024 Xiaomi Corporation";
      return HelpText.DefaultParsingErrorsHandler(result, h);
    }, e => e);
    Console.WriteLine(helpText);
  }

  private static void Run(Options options)
  {

    // Validate CLI requirements
    bool useZipvoice = !string.IsNullOrEmpty(options.PromptAudio);
    if (useZipvoice)
    {
      var missing = new List<string>();
      if (string.IsNullOrWhiteSpace(options.ZipvoiceTextModel)) missing.Add("--zipvoice-text-model");
      if (string.IsNullOrWhiteSpace(options.ZipvoiceFlowMatchingModel)) missing.Add("--zipvoice-flow-matching-model");
      if (string.IsNullOrWhiteSpace(options.ZipvoiceVocoder)) missing.Add("--zipvoice-vocoder");
      if (string.IsNullOrWhiteSpace(options.PromptText)) missing.Add("--prompt-text");
      if (missing.Count > 0)
      {
        Console.Error.WriteLine($"Missing required options for ZipVoice: {string.Join(", ", missing)}");
        return;
      }
    }

    var config = new OfflineTtsConfig();

    config.Model.Vits.Model = options.Model;
    config.Model.Vits.Lexicon = options.Lexicon;
    config.Model.Vits.Tokens = options.Tokens;
    config.Model.Vits.DataDir = options.DataDir;
    config.Model.Vits.NoiseScale = options.NoiseScale;
    config.Model.Vits.NoiseScaleW = options.NoiseScaleW;
    config.Model.Vits.LengthScale = options.LengthScale;

    config.Model.Matcha.AcousticModel = options.AcousticModel;
    config.Model.Matcha.Vocoder = options.Vocoder;
    config.Model.Matcha.Lexicon = options.Lexicon;
    config.Model.Matcha.Tokens = options.Tokens;
    config.Model.Matcha.DataDir = options.DataDir;
    config.Model.Matcha.NoiseScale = options.NoiseScale;
    config.Model.Matcha.LengthScale = options.LengthScale;

    config.Model.Zipvoice.Tokens = options.Tokens;
    config.Model.Zipvoice.TextModel = options.ZipvoiceTextModel;
    config.Model.Zipvoice.FlowMatchingModel = options.ZipvoiceFlowMatchingModel;
    config.Model.Zipvoice.DataDir = options.ZipvoiceDataDir;
    config.Model.Zipvoice.PinyinDict = options.ZipvoicePinyinDict;
    config.Model.Zipvoice.Vocoder = options.ZipvoiceVocoder;
    config.Model.Zipvoice.FeatScale = options.ZipvoiceFeatScale;
    config.Model.Zipvoice.TShift = options.ZipvoiceTShift;
    config.Model.Zipvoice.TargetRms = options.ZipvoiceTargetRms;
    config.Model.Zipvoice.GuidanceScale = options.ZipvoiceGuidanceScale;

    config.Model.NumThreads = 1;
    config.Model.Debug = options.Debug;
    config.Model.Provider = "cpu";
    config.RuleFsts = options.RuleFsts;
    config.MaxNumSentences = options.MaxNumSentences;

    var tts = new OfflineTts(config);
    var speed = 1.0f / options.LengthScale;
    var sid = options.SpeakerId;

    Console.WriteLine(PortAudio.VersionInfo.versionText);
    PortAudio.Initialize();
    Console.WriteLine($"Number of devices: {PortAudio.DeviceCount}");

    for (int i = 0; i != PortAudio.DeviceCount; ++i)
    {
      Console.WriteLine($" Device {i}");
      DeviceInfo deviceInfo = PortAudio.GetDeviceInfo(i);
      Console.WriteLine($"   Name: {deviceInfo.name}");
      Console.WriteLine($"   Max output channels: {deviceInfo.maxOutputChannels}");
      Console.WriteLine($"   Default sample rate: {deviceInfo.defaultSampleRate}");
    }
    int deviceIndex = PortAudio.DefaultOutputDevice;
    if (deviceIndex == PortAudio.NoDevice)
    {
      Console.WriteLine("No default output device found. Please use ../offline-tts instead");
      Environment.Exit(1);
    }

    var info = PortAudio.GetDeviceInfo(deviceIndex);
    Console.WriteLine();
    Console.WriteLine($"Use output default device {deviceIndex} ({info.name})");

    var param = new StreamParameters();
    param.device = deviceIndex;
    param.channelCount = 1;
    param.sampleFormat = SampleFormat.Float32;
    param.suggestedLatency = info.defaultLowOutputLatency;
    param.hostApiSpecificStreamInfo = IntPtr.Zero;

    // https://learn.microsoft.com/en-us/dotnet/standard/collections/thread-safe/blockingcollection-overview
    var dataItems = new BlockingCollection<float[]>();

    var MyCallback = (IntPtr samples, int n) =>
    {
      float[] data = new float[n];

      Marshal.Copy(samples, data, 0, n);

      dataItems.Add(data);

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var playFinished = false;

    float[]? lastSampleArray = null;
    int lastIndex = 0; // not played

    PortAudioSharp.Stream.Callback playCallback = (IntPtr input, IntPtr output,
        UInt32 frameCount,
        ref StreamCallbackTimeInfo timeInfo,
        StreamCallbackFlags statusFlags,
        IntPtr userData
        ) =>
    {
      if (dataItems.IsCompleted && lastSampleArray == null && lastIndex == 0)
      {
        Console.WriteLine($"Finished playing");
        playFinished = true;
        return StreamCallbackResult.Complete;
      }

      int expected = Convert.ToInt32(frameCount);
      int i = 0;

      while ((lastSampleArray != null || dataItems.Count != 0) && (i < expected))
      {
        int needed = expected - i;

        if (lastSampleArray != null)
        {
          int remaining = lastSampleArray.Length - lastIndex;
          if (remaining >= needed)
          {
            float[] this_block = lastSampleArray.Skip(lastIndex).Take(needed).ToArray();
            lastIndex += needed;
            if (lastIndex == lastSampleArray.Length)
            {
              lastSampleArray = null;
              lastIndex = 0;
            }

            Marshal.Copy(this_block, 0, IntPtr.Add(output, i * sizeof(float)), needed);
            return StreamCallbackResult.Continue;
          }

          float[] this_block2 = lastSampleArray.Skip(lastIndex).Take(remaining).ToArray();
          lastIndex = 0;
          lastSampleArray = null;

          Marshal.Copy(this_block2, 0, IntPtr.Add(output, i * sizeof(float)), remaining);
          i += remaining;
          continue;
        }

        if (dataItems.Count != 0)
        {
          lastSampleArray = dataItems.Take();
          lastIndex = 0;
        }
      }

      if (i < expected)
      {
        int sizeInBytes = (expected - i) * 4;
        Marshal.Copy(new byte[sizeInBytes], 0, IntPtr.Add(output, i * sizeof(float)), sizeInBytes);
      }

      return StreamCallbackResult.Continue;
    };

    PortAudioSharp.Stream stream = new PortAudioSharp.Stream(inParams: null, outParams: param, sampleRate: tts.SampleRate,
        framesPerBuffer: 0,
        streamFlags: StreamFlags.ClipOff,
        callback: playCallback,
        userData: IntPtr.Zero
        );

    stream.Start();

    var callback = new OfflineTtsCallback(MyCallback);

    if (useZipvoice)
    {
      if (string.IsNullOrWhiteSpace(options.PromptText))
      {
        Console.Error.WriteLine("For ZipVoice zero-shot TTS, --prompt-text is required when --prompt-audio is provided");
        dataItems.CompleteAdding();
        while (!playFinished)
        {
          Thread.Sleep(100);
        }
        return;
      }

      // Zero-shot cloning path (ZipVoice): generate first, then play
      var promptReader = new WaveReader(options.PromptAudio);
      var audio = tts.GenerateWithZipvoice(
          options.Text,
          options.PromptText,
          promptReader.Samples,
          promptReader.SampleRate,
          speed,
          options.ZipvoiceNumSteps);

      var ok = audio.SaveToWaveFile(options.OutputFilename);

      if (ok)
      {
        Console.WriteLine($"Wrote to {options.OutputFilename} succeeded!");
      }
      else
      {
        Console.WriteLine($"Failed to write {options.OutputFilename}");
      }

      // Enqueue the generated waveform for playback in chunks
      var outReader = new WaveReader(options.OutputFilename);
      const int chunk = 4096;
      int offset = 0;
      while (offset < outReader.Samples.Length)
      {
        int n = Math.Min(chunk, outReader.Samples.Length - offset);
        float[] slice = outReader.Samples.Skip(offset).Take(n).ToArray();
        dataItems.Add(slice);
        offset += n;
      }
      dataItems.CompleteAdding();
    }
    else
    {

      var audio = tts.GenerateWithCallback(options.Text, speed, sid, callback);
      var ok = audio.SaveToWaveFile(options.OutputFilename);

      if (ok)
      {
        Console.WriteLine($"Wrote to {options.OutputFilename} succeeded!");
      }
      else
      {
        Console.WriteLine($"Failed to write {options.OutputFilename}");
      }
      dataItems.CompleteAdding();

    }

    while (!playFinished)
    {
      Thread.Sleep(100); // 100ms
    }
  }
}
