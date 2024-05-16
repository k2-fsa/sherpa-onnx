/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineTtsVitsModelConfig
  {
    public OfflineTtsVitsModelConfig()
    {
      Model = "";
      Lexicon = "";
      Tokens = "";
      DataDir = "";

      NoiseScale = 0.667F;
      NoiseScaleW = 0.8F;
      LengthScale = 1.0F;

      DictDir = "";
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Lexicon;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Tokens;

    [MarshalAs(UnmanagedType.LPStr)]
    public string DataDir;

    public float NoiseScale;
    public float NoiseScaleW;
    public float LengthScale;

    [MarshalAs(UnmanagedType.LPStr)]
    public string DictDir;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineTtsModelConfig
  {
    public OfflineTtsModelConfig()
    {
      Vits = new OfflineTtsVitsModelConfig();
      NumThreads = 1;
      Debug = 0;
      Provider = "cpu";
    }

    public OfflineTtsVitsModelConfig Vits;
    public int NumThreads;
    public int Debug;
    [MarshalAs(UnmanagedType.LPStr)]
    public string Provider;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineTtsConfig
  {
    public OfflineTtsConfig()
    {
      Model = new OfflineTtsModelConfig();
      RuleFsts = "";
      MaxNumSentences = 1;
      RuleFars = "";
    }
    public OfflineTtsModelConfig Model;

    [MarshalAs(UnmanagedType.LPStr)]
    public string RuleFsts;

    public int MaxNumSentences;

    [MarshalAs(UnmanagedType.LPStr)]
    public string RuleFars;
  }

  public class OfflineTtsGeneratedAudio
  {
    public OfflineTtsGeneratedAudio(IntPtr p)
    {
      _handle = new HandleRef(this, p);
    }

    public bool SaveToWaveFile(String filename)
    {
      Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
      int status = SherpaOnnxWriteWave(impl.Samples, impl.NumSamples, impl.SampleRate, filename);
      return status == 1;
    }

    ~OfflineTtsGeneratedAudio()
    {
      Cleanup();
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    private void Cleanup()
    {
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Impl
    {
      public IntPtr Samples;
      public int NumSamples;
      public int SampleRate;
    }

    private HandleRef _handle;
    public IntPtr Handle => _handle.Handle;

    public int NumSamples
    {
      get
      {
        Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
        return impl.NumSamples;
      }
    }

    public int SampleRate
    {
      get
      {
        Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
        return impl.SampleRate;
      }
    }

    public float[] Samples
    {
      get
      {
        Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));

        float[] samples = new float[impl.NumSamples];
        Marshal.Copy(impl.Samples, samples, 0, impl.NumSamples);
        return samples;
      }
    }

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroyOfflineTtsGeneratedAudio(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxWriteWave(IntPtr samples, int n, int sample_rate, [MarshalAs(UnmanagedType.LPStr)] string filename);
  }

  // IntPtr is actuallly a `const float*` from C++
  public delegate void OfflineTtsCallback(IntPtr samples, int n);

  public class OfflineTts : IDisposable
  {
    public OfflineTts(OfflineTtsConfig config)
    {
      IntPtr h = SherpaOnnxCreateOfflineTts(ref config);
      _handle = new HandleRef(this, h);
    }

    public OfflineTtsGeneratedAudio Generate(String text, float speed, int speakerId)
    {
      IntPtr p = SherpaOnnxOfflineTtsGenerate(_handle.Handle, text, speakerId, speed);
      return new OfflineTtsGeneratedAudio(p);
    }

    public OfflineTtsGeneratedAudio GenerateWithCallback(String text, float speed, int speakerId, OfflineTtsCallback callback)
    {
      IntPtr p = SherpaOnnxOfflineTtsGenerateWithCallback(_handle.Handle, text, speakerId, speed, callback);
      return new OfflineTtsGeneratedAudio(p);
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~OfflineTts()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      SherpaOnnxDestroyOfflineTts(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    public int SampleRate
    {
      get
      {
        return SherpaOnnxOfflineTtsSampleRate(_handle.Handle);
      }
    }

    public int NumSpeakers
    {
      get
      {
        return SherpaOnnxOfflineTtsNumSpeakers(_handle.Handle);
      }
    }

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateOfflineTts(ref OfflineTtsConfig config);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroyOfflineTts(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxOfflineTtsSampleRate(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxOfflineTtsNumSpeakers(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxOfflineTtsGenerate(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string text, int sid, float speed);

    [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr SherpaOnnxOfflineTtsGenerateWithCallback(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string text, int sid, float speed, OfflineTtsCallback callback);
  }



  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineTransducerModelConfig
  {
    public OfflineTransducerModelConfig()
    {
      Encoder = "";
      Decoder = "";
      Joiner = "";
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Encoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Decoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Joiner;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineParaformerModelConfig
  {
    public OfflineParaformerModelConfig()
    {
      Model = "";
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineNemoEncDecCtcModelConfig
  {
    public OfflineNemoEncDecCtcModelConfig()
    {
      Model = "";
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineWhisperModelConfig
  {
    public OfflineWhisperModelConfig()
    {
      Encoder = "";
      Decoder = "";
      Language = "";
      Task = "transcribe";
      TailPaddings = -1;
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Encoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Decoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Language;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Task;

    public int TailPaddings;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineTdnnModelConfig
  {
    public OfflineTdnnModelConfig()
    {
      Model = "";
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineLMConfig
  {
    public OfflineLMConfig()
    {
      Model = "";
      Scale = 0.5F;
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;

    public float Scale;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineModelConfig
  {
    public OfflineModelConfig()
    {
      Transducer = new OfflineTransducerModelConfig();
      Paraformer = new OfflineParaformerModelConfig();
      NeMoCtc = new OfflineNemoEncDecCtcModelConfig();
      Whisper = new OfflineWhisperModelConfig();
      Tdnn = new OfflineTdnnModelConfig();
      Tokens = "";
      NumThreads = 1;
      Debug = 0;
      Provider = "cpu";
      ModelType = "";
    }
    public OfflineTransducerModelConfig Transducer;
    public OfflineParaformerModelConfig Paraformer;
    public OfflineNemoEncDecCtcModelConfig NeMoCtc;
    public OfflineWhisperModelConfig Whisper;
    public OfflineTdnnModelConfig Tdnn;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Tokens;

    public int NumThreads;

    public int Debug;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Provider;

    [MarshalAs(UnmanagedType.LPStr)]
    public string ModelType;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OfflineRecognizerConfig
  {
    public OfflineRecognizerConfig()
    {
      FeatConfig = new FeatureConfig();
      ModelConfig = new OfflineModelConfig();
      LmConfig = new OfflineLMConfig();

      DecodingMethod = "greedy_search";
      MaxActivePaths = 4;
      HotwordsFile = "";
      HotwordsScore = 1.5F;

    }
    public FeatureConfig FeatConfig;
    public OfflineModelConfig ModelConfig;
    public OfflineLMConfig LmConfig;

    [MarshalAs(UnmanagedType.LPStr)]
    public string DecodingMethod;

    public int MaxActivePaths;

    [MarshalAs(UnmanagedType.LPStr)]
    public string HotwordsFile;

    public float HotwordsScore;
  }

  public class OfflineRecognizerResult
  {
    public OfflineRecognizerResult(IntPtr handle)
    {
      Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

      // PtrToStringUTF8() requires .net standard 2.1
      // _text = Marshal.PtrToStringUTF8(impl.Text);

      int length = 0;

      unsafe
      {
        byte* buffer = (byte*)impl.Text;
        while (*buffer != 0)
        {
          ++buffer;
          length += 1;
        }
      }

      byte[] stringBuffer = new byte[length];
      Marshal.Copy(impl.Text, stringBuffer, 0, length);
      _text = Encoding.UTF8.GetString(stringBuffer);
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Impl
    {
      public IntPtr Text;
    }

    private String _text;
    public String Text => _text;
  }

  public class OfflineStream : IDisposable
  {
    public OfflineStream(IntPtr p)
    {
      _handle = new HandleRef(this, p);
    }

    public void AcceptWaveform(int sampleRate, float[] samples)
    {
      AcceptWaveform(Handle, sampleRate, samples, samples.Length);
    }

    public OfflineRecognizerResult Result
    {
      get
      {
        IntPtr h = GetResult(_handle.Handle);
        OfflineRecognizerResult result = new OfflineRecognizerResult(h);
        DestroyResult(h);
        return result;
      }
    }

    ~OfflineStream()
    {
      Cleanup();
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    private void Cleanup()
    {
      DestroyOfflineStream(Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;
    public IntPtr Handle => _handle.Handle;

    [DllImport(Dll.Filename)]
    private static extern void DestroyOfflineStream(IntPtr handle);

    [DllImport(Dll.Filename, EntryPoint = "AcceptWaveformOffline")]
    private static extern void AcceptWaveform(IntPtr handle, int sampleRate, float[] samples, int n);

    [DllImport(Dll.Filename, EntryPoint = "GetOfflineStreamResult")]
    private static extern IntPtr GetResult(IntPtr handle);

    [DllImport(Dll.Filename, EntryPoint = "DestroyOfflineRecognizerResult")]
    private static extern void DestroyResult(IntPtr handle);
  }

  public class OfflineRecognizer : IDisposable
  {
    public OfflineRecognizer(OfflineRecognizerConfig config)
    {
      IntPtr h = CreateOfflineRecognizer(ref config);
      _handle = new HandleRef(this, h);
    }

    public OfflineStream CreateStream()
    {
      IntPtr p = CreateOfflineStream(_handle.Handle);
      return new OfflineStream(p);
    }

    public void Decode(OfflineStream stream)
    {
      Decode(_handle.Handle, stream.Handle);
    }

    // The caller should ensure all passed streams are ready for decoding.
    public void Decode(IEnumerable<OfflineStream> streams)
    {
      IntPtr[] ptrs = streams.Select(s => s.Handle).ToArray();
      Decode(_handle.Handle, ptrs, ptrs.Length);
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~OfflineRecognizer()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      DestroyOfflineRecognizer(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    [DllImport(Dll.Filename)]
    private static extern IntPtr CreateOfflineRecognizer(ref OfflineRecognizerConfig config);

    [DllImport(Dll.Filename)]
    private static extern void DestroyOfflineRecognizer(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr CreateOfflineStream(IntPtr handle);

    [DllImport(Dll.Filename, EntryPoint = "DecodeOfflineStream")]
    private static extern void Decode(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename, EntryPoint = "DecodeMultipleOfflineStreams")]
    private static extern void Decode(IntPtr handle, IntPtr[] streams, int n);
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct SpeakerEmbeddingExtractorConfig
  {
    public SpeakerEmbeddingExtractorConfig()
    {
      Model = "";
      NumThreads = 1;
      Debug = 0;
      Provider = "cpu";
    }

    [MarshalAs(UnmanagedType.LPStr)]
    public string Model;

    public int NumThreads;
    public int Debug;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Provider;
  }

  public class SpeakerEmbeddingExtractor : IDisposable
  {
    public SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig config)
    {
      IntPtr h = SherpaOnnxCreateSpeakerEmbeddingExtractor(ref config);
      _handle = new HandleRef(this, h);
    }

    public OnlineStream CreateStream()
    {
      IntPtr p = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(_handle.Handle);
      return new OnlineStream(p);
    }

    public bool IsReady(OnlineStream stream)
    {
      return SherpaOnnxSpeakerEmbeddingExtractorIsReady(_handle.Handle, stream.Handle) != 0;
    }

    public float[] Compute(OnlineStream stream)
    {
      IntPtr p = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(_handle.Handle, stream.Handle);

      int dim = Dim;
      float[] ans = new float[dim];
      Marshal.Copy(p, ans, 0, dim);

      SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p);

      return ans;
    }

    public int Dim
    {
      get
      {
        return SherpaOnnxSpeakerEmbeddingExtractorDim(_handle.Handle);
      }
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~SpeakerEmbeddingExtractor()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      SherpaOnnxDestroySpeakerEmbeddingExtractor(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateSpeakerEmbeddingExtractor(ref SpeakerEmbeddingExtractorConfig config);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpeakerEmbeddingExtractor(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingExtractorDim(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpeakerEmbeddingExtractorCreateStream(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingExtractorIsReady(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(IntPtr p);
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct SpokenLanguageIdentificationWhisperConfig
  {
    public SpokenLanguageIdentificationWhisperConfig()
    {
      Encoder = "";
      Decoder = "";
      TailPaddings = -1;
    }

    [MarshalAs(UnmanagedType.LPStr)]
    public string Encoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Decoder;

    public int TailPaddings;
  }

  public struct SpokenLanguageIdentificationConfig
  {
    public SpokenLanguageIdentificationConfig()
    {
      Whisper = new SpokenLanguageIdentificationWhisperConfig();
      NumThreads = 1;
      Debug = 0;
      Provider = "cpu";
    }
    public SpokenLanguageIdentificationWhisperConfig Whisper;

    public int NumThreads;
    public int Debug;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Provider;
  }

  public class SpeakerEmbeddingManager : IDisposable
  {
    public SpeakerEmbeddingManager(int dim)
    {
      IntPtr h = SherpaOnnxCreateSpeakerEmbeddingManager(dim);
      _handle = new HandleRef(this, h);
      this._dim = dim;
    }

    public bool Add(string name, float[] v)
    {
      return SherpaOnnxSpeakerEmbeddingManagerAdd(_handle.Handle, name, v) == 1;
    }

    public bool Add(string name, ICollection<float[]> v_list)
    {
      int n = v_list.Count;
      float[] v = new float[n * _dim];
      int i = 0;
      foreach (var item in v_list)
      {
        item.CopyTo(v, i);
        i += _dim;
      }

      return SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(_handle.Handle, name, v, n) == 1;
    }

    public bool Remove(string name)
    {
      return SherpaOnnxSpeakerEmbeddingManagerRemove(_handle.Handle, name) == 1;
    }

    public string Search(float[] v, float threshold)
    {
      IntPtr p = SherpaOnnxSpeakerEmbeddingManagerSearch(_handle.Handle, v, threshold);

      string s = "";
      int length = 0;

      unsafe
      {
        byte* b = (byte*)p;
        if (b != null)
        {
          while (*b != 0)
          {
            ++b;
            length += 1;
          }
        }
      }

      if (length > 0)
      {
        byte[] stringBuffer = new byte[length];
        Marshal.Copy(p, stringBuffer, 0, length);
        s = Encoding.UTF8.GetString(stringBuffer);
      }

      SherpaOnnxSpeakerEmbeddingManagerFreeSearch(p);

      return s;
    }

    public bool Verify(string name, float[] v, float threshold)
    {
      return SherpaOnnxSpeakerEmbeddingManagerVerify(_handle.Handle, name, v, threshold) == 1;
    }

    public bool Contains(string name)
    {
      return SherpaOnnxSpeakerEmbeddingManagerContains(_handle.Handle, name) == 1;
    }

    public string[] GetAllSpeakers()
    {
      if (NumSpeakers == 0)
      {
        return new string[] { };
      }

      IntPtr names = SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(_handle.Handle);

      string[] ans = new string[NumSpeakers];

      unsafe
      {
        byte** p = (byte**)names;
        for (int i = 0; i != NumSpeakers; i++)
        {
          int length = 0;
          byte* s = p[i];
          while (*s != 0)
          {
            ++s;
            length += 1;
          }
          byte[] stringBuffer = new byte[length];
          Marshal.Copy((IntPtr)p[i], stringBuffer, 0, length);
          ans[i] = Encoding.UTF8.GetString(stringBuffer);
        }
      }

      SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(names);

      return ans;
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~SpeakerEmbeddingManager()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      SherpaOnnxDestroySpeakerEmbeddingManager(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    public int NumSpeakers
    {
      get
      {
        return SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(_handle.Handle);
      }
    }

    private HandleRef _handle;
    private int _dim;


    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateSpeakerEmbeddingManager(int dim);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpeakerEmbeddingManager(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerAdd(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v, int n);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerRemove(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpeakerEmbeddingManagerSearch(IntPtr handle, float[] v, float threshold);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxSpeakerEmbeddingManagerFreeSearch(IntPtr p);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerVerify(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name, float[] v, float threshold);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerContains(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(IntPtr names);
  }

  public class SpokenLanguageIdentificationResult
  {
    public SpokenLanguageIdentificationResult(IntPtr handle)
    {
      Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

      // PtrToStringUTF8() requires .net standard 2.1
      // _text = Marshal.PtrToStringUTF8(impl.Text);

      int length = 0;

      unsafe
      {
        byte* buffer = (byte*)impl.Lang;
        while (*buffer != 0)
        {
          ++buffer;
          length += 1;
        }
      }

      byte[] stringBuffer = new byte[length];
      Marshal.Copy(impl.Lang, stringBuffer, 0, length);
      _lang = Encoding.UTF8.GetString(stringBuffer);
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Impl
    {
      public IntPtr Lang;
    }

    private String _lang;
    public String Lang => _lang;
  }

  public class SpokenLanguageIdentification : IDisposable
  {
    public SpokenLanguageIdentification(SpokenLanguageIdentificationConfig config)
    {
      IntPtr h = SherpaOnnxCreateSpokenLanguageIdentification(ref config);
      _handle = new HandleRef(this, h);
    }

    public OfflineStream CreateStream()
    {
      IntPtr p = SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(_handle.Handle);
      return new OfflineStream(p);
    }

    public SpokenLanguageIdentificationResult Compute(OfflineStream stream)
    {
      IntPtr h = SherpaOnnxSpokenLanguageIdentificationCompute(_handle.Handle, stream.Handle);
      SpokenLanguageIdentificationResult result = new SpokenLanguageIdentificationResult(h);
      SherpaOnnxDestroySpokenLanguageIdentificationResult(h);
      return result;
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~SpokenLanguageIdentification()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      SherpaOnnxDestroySpokenLanguageIdentification(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateSpokenLanguageIdentification(ref SpokenLanguageIdentificationConfig config);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpokenLanguageIdentification(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxSpokenLanguageIdentificationCompute(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroySpokenLanguageIdentificationResult(IntPtr handle);
  }
}
