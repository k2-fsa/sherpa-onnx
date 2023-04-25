using System.Runtime.InteropServices;
using System;

namespace SherpaOnnx {

[StructLayout(LayoutKind.Sequential)]
public struct OnlineTransducerModelConfig {
  [MarshalAs(UnmanagedType.LPStr)]
  public string Encoder;

  [MarshalAs(UnmanagedType.LPStr)]
  public string Decoder;

  [MarshalAs(UnmanagedType.LPStr)]
  public string Joiner;

  [MarshalAs(UnmanagedType.LPStr)]
  public string Tokens;

  public int NumThreads;
  public int Debug;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeatureConfig {
  public int SampleRate;
  public int FeatureDim;
}

[StructLayout(LayoutKind.Sequential)]
public struct OnlineRecognizerConfig {
  public FeatureConfig FeatConfig;
  public OnlineTransducerModelConfig ModelConfig;

  [MarshalAs(UnmanagedType.LPStr)]
  public string DecodingMethod;

  public int MaxActivePaths;
  public int EnableEndpoint;
  public float Rule1MinTrailingSilence;
  public float Rule2MinTrailingSilence;
  public float Rule3MinUtteranceLength;
}

// please see
// https://learn.microsoft.com/en-us/dotnet/api/system.idisposable.dispose?view=net-7.0
public class OnlineRecognizer : IDisposable {
  public OnlineRecognizer(OnlineRecognizerConfig config) {
    handle = CreateOnlineRecognizer(config);
  }

  public OnlineStream CreateStream() {
    IntPtr p = CreateOnlineStream(handle);
    return new OnlineStream(p);
  }

  public void Dispose() {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    if(!this.disposed) {
      DestroyOnlineRecognizer(handle);
      handle = IntPtr.Zero;
      disposed = true;
    }
  }

  ~OnlineRecognizer() {
    Dispose(disposing: false);
  }

  private IntPtr handle;
  private bool disposed = false;

  private const string dllName = "sherpa-onnx-c-api.dll";

  [DllImport(dllName)]
  public static extern IntPtr CreateOnlineRecognizer(OnlineRecognizerConfig config);

  [DllImport(dllName)]
  public static extern void DestroyOnlineRecognizer(IntPtr handle);

  [DllImport(dllName)]
  public static extern IntPtr CreateOnlineStream(IntPtr handle);
}

public class OnlineStream : IDisposable {
  public OnlineStream(IntPtr ptr) {
    this.handle = ptr;
  }

  public void Dispose() {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    if(!this.disposed) {
      DestoryOnlineStream(handle);
      handle = IntPtr.Zero;
      disposed = true;
    }
  }

  ~OnlineStream() {
    Dispose(disposing: false);
  }

  private IntPtr handle;
  private bool disposed = false;

  private const string dllName = "sherpa-onnx-c-api.dll";

  [DllImport(dllName)]
  public static extern void DestoryOnlineStream(IntPtr handle);
}



class Hello {
  public static void Main(string[] args) {
    OnlineRecognizerConfig config = new OnlineRecognizerConfig();
    config.FeatConfig.SampleRate = 16000;
    config.FeatConfig.FeatureDim = 80;

    config.ModelConfig.Encoder = "encoder-epoch-99-avg-1.onnx";
    config.ModelConfig.Decoder = "decoder-epoch-99-avg-1.onnx";
    config.ModelConfig.Joiner = "joiner-epoch-99-avg-1.onnx";
    config.ModelConfig.Tokens = "tokens.txt";
    config.ModelConfig.NumThreads = 2;
    config.ModelConfig.Debug = 1;

    config.DecodingMethod = "greedy_search";
    config.MaxActivePaths = 4;
    config.EnableEndpoint = 1;
    config.Rule1MinTrailingSilence = 1.2F;
    config.Rule2MinTrailingSilence = 2.0F;
    config.Rule3MinUtteranceLength = 20.0F;

    OnlineRecognizer recognizer = new OnlineRecognizer(config);
    OnlineStream stream = recognizer.CreateStream();
    Console.WriteLine("Hello world");
  }
}

}
