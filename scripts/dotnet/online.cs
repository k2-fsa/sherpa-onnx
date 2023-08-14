/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes

using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
  internal static class Dll
  {
    public const string Filename = "sherpa-onnx-c-api";
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OnlineTransducerModelConfig
  {
    public OnlineTransducerModelConfig()
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
  public struct OnlineParaformerModelConfig
  {
    public OnlineParaformerModelConfig()
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
  public struct OnlineModelConfig
  {
    public OnlineModelConfig()
    {
      Transducer = new OnlineTransducerModelConfig();
      Paraformer = new OnlineParaformerModelConfig();
      Tokens = "";
      NumThreads = 1;
      Provider = "cpu";
      Debug = 0;
      ModelType = "";
    }

    public OnlineTransducerModelConfig Transducer;
    public OnlineParaformerModelConfig Paraformer;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Tokens;

    /// Number of threads used to run the neural network model
    public int NumThreads;

    [MarshalAs(UnmanagedType.LPStr)]
    public string Provider;

    /// true to print debug information of the model
    public int Debug;

    [MarshalAs(UnmanagedType.LPStr)]
    public string ModelType;
  }

  /// It expects 16 kHz 16-bit single channel wave format.
  [StructLayout(LayoutKind.Sequential)]
  public struct FeatureConfig
  {
    public FeatureConfig()
    {
      SampleRate = 16000;
      FeatureDim = 80;
    }
    /// Sample rate of the input data. MUST match the one expected
    /// by the model. For instance, it should be 16000 for models provided
    /// by us.
    public int SampleRate;

    /// Feature dimension of the model.
    /// For instance, it should be 80 for models provided by us.
    public int FeatureDim;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct OnlineRecognizerConfig
  {
    public OnlineRecognizerConfig()
    {
      FeatConfig = new FeatureConfig();
      ModelConfig = new OnlineModelConfig();
      DecodingMethod = "greedy_search";
      MaxActivePaths = 4;
      EnableEndpoint = 0;
      Rule1MinTrailingSilence = 1.2F;
      Rule2MinTrailingSilence = 2.4F;
      Rule3MinUtteranceLength = 20.0F;
    }
    public FeatureConfig FeatConfig;
    public OnlineModelConfig ModelConfig;

    [MarshalAs(UnmanagedType.LPStr)]
    public string DecodingMethod;

    /// Used only when decoding_method is modified_beam_search
    /// Example value: 4
    public int MaxActivePaths;

    /// 0 to disable endpoint detection.
    /// A non-zero value to enable endpoint detection.
    public int EnableEndpoint;

    /// An endpoint is detected if trailing silence in seconds is larger than
    /// this value even if nothing has been decoded.
    /// Used only when enable_endpoint is not 0.
    public float Rule1MinTrailingSilence;

    /// An endpoint is detected if trailing silence in seconds is larger than
    /// this value after something that is not blank has been decoded.
    /// Used only when enable_endpoint is not 0.
    public float Rule2MinTrailingSilence;

    /// An endpoint is detected if the utterance in seconds is larger than
    /// this value.
    /// Used only when enable_endpoint is not 0.
    public float Rule3MinUtteranceLength;
  }

  public class OnlineRecognizerResult
  {
    public OnlineRecognizerResult(IntPtr handle)
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

  public class OnlineStream : IDisposable
  {
    public OnlineStream(IntPtr p)
    {
      _handle = new HandleRef(this, p);
    }

    public void AcceptWaveform(int sampleRate, float[] samples)
    {
      AcceptWaveform(Handle, sampleRate, samples, samples.Length);
    }

    public void InputFinished()
    {
      InputFinished(Handle);
    }

    ~OnlineStream()
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
      DestroyOnlineStream(Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;
    public IntPtr Handle => _handle.Handle;

    [DllImport(Dll.Filename)]
    private static extern void DestroyOnlineStream(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern void AcceptWaveform(IntPtr handle, int sampleRate, float[] samples, int n);

    [DllImport(Dll.Filename)]
    private static extern void InputFinished(IntPtr handle);
  }

  // please see
  // https://www.mono-project.com/docs/advanced/pinvoke/#gc-safe-pinvoke-code
  // https://www.mono-project.com/docs/advanced/pinvoke/#properly-disposing-of-resources
  public class OnlineRecognizer : IDisposable
  {
    public OnlineRecognizer(OnlineRecognizerConfig config)
    {
      IntPtr h = CreateOnlineRecognizer(ref config);
      _handle = new HandleRef(this, h);
    }

    public OnlineStream CreateStream()
    {
      IntPtr p = CreateOnlineStream(_handle.Handle);
      return new OnlineStream(p);
    }

    /// Return true if the passed stream is ready for decoding.
    public bool IsReady(OnlineStream stream)
    {
      return IsReady(_handle.Handle, stream.Handle) != 0;
    }

    /// Return true if an endpoint is detected for this stream.
    /// You probably need to invoke Reset(stream) when this method returns
    /// true.
    public bool IsEndpoint(OnlineStream stream)
    {
      return IsEndpoint(_handle.Handle, stream.Handle) != 0;
    }

    /// You have to ensure that IsReady(stream) returns true before
    /// you call this method
    public void Decode(OnlineStream stream)
    {
      Decode(_handle.Handle, stream.Handle);
    }

    // The caller should ensure all passed streams are ready for decoding.
    public void Decode(IEnumerable<OnlineStream> streams)
    {
      IntPtr[] ptrs = streams.Select(s => s.Handle).ToArray();
      Decode(_handle.Handle, ptrs, ptrs.Length);
    }

    public OnlineRecognizerResult GetResult(OnlineStream stream)
    {
      IntPtr h = GetResult(_handle.Handle, stream.Handle);
      OnlineRecognizerResult result = new OnlineRecognizerResult(h);
      DestroyResult(h);
      return result;
    }

    /// When this method returns, IsEndpoint(stream) will return false.
    public void Reset(OnlineStream stream)
    {
      Reset(_handle.Handle, stream.Handle);
    }

    public void Dispose()
    {
      Cleanup();
      // Prevent the object from being placed on the
      // finalization queue
      System.GC.SuppressFinalize(this);
    }

    ~OnlineRecognizer()
    {
      Cleanup();
    }

    private void Cleanup()
    {
      DestroyOnlineRecognizer(_handle.Handle);

      // Don't permit the handle to be used again.
      _handle = new HandleRef(this, IntPtr.Zero);
    }

    private HandleRef _handle;

    [DllImport(Dll.Filename)]
    private static extern IntPtr CreateOnlineRecognizer(ref OnlineRecognizerConfig config);

    [DllImport(Dll.Filename)]
    private static extern void DestroyOnlineRecognizer(IntPtr handle);

    [DllImport(Dll.Filename)]
    private static extern IntPtr CreateOnlineStream(IntPtr handle);

    [DllImport(Dll.Filename, EntryPoint = "IsOnlineStreamReady")]
    private static extern int IsReady(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename, EntryPoint = "DecodeOnlineStream")]
    private static extern void Decode(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename, EntryPoint = "DecodeMultipleOnlineStreams")]
    private static extern void Decode(IntPtr handle, IntPtr[] streams, int n);

    [DllImport(Dll.Filename, EntryPoint = "GetOnlineStreamResult")]
    private static extern IntPtr GetResult(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename, EntryPoint = "DestroyOnlineRecognizerResult")]
    private static extern void DestroyResult(IntPtr result);

    [DllImport(Dll.Filename)]
    private static extern void Reset(IntPtr handle, IntPtr stream);

    [DllImport(Dll.Filename)]
    private static extern int IsEndpoint(IntPtr handle, IntPtr stream);
  }
}
