/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

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
            ModelingUnit = "cjkchar";
            BpeVocab = "";
            TeleSpeechCtc = "";
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

        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelingUnit;

        [MarshalAs(UnmanagedType.LPStr)]
        public string BpeVocab;

        [MarshalAs(UnmanagedType.LPStr)]
        public string TeleSpeechCtc;
    }


}