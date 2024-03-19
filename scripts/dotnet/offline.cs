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
    }
    public OfflineTtsModelConfig Model;

    [MarshalAs(UnmanagedType.LPStr)]
    public string RuleFsts;

    public int MaxNumSentences;
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

    [DllImport(Dll.Filename)]
    private static extern IntPtr SherpaOnnxCreateOfflineTts(ref OfflineTtsConfig config);

    [DllImport(Dll.Filename)]
    private static extern void SherpaOnnxDestroyOfflineTts(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxOfflineTtsSampleRate(IntPtr handle);

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
    }
    [MarshalAs(UnmanagedType.LPStr)]
    public string Encoder;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Decoder;
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
        }
        length = (int)(buffer - (byte*)impl.Text);
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

    /// You have to ensure that IsReady(stream) returns true before
    /// you call this method
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

}
