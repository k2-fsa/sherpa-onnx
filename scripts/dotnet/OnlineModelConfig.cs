/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OnlineModelConfig
    {
        public OnlineModelConfig()
        {
            Transducer = new OnlineTransducerModelConfig();
            Paraformer = new OnlineParaformerModelConfig();
            Zipformer2Ctc = new OnlineZipformer2CtcModelConfig();
            Tokens = "";
            NumThreads = 1;
            Provider = "cpu";
            Debug = 0;
            ModelType = "";
            ModelingUnit = "cjkchar";
            BpeVocab = "";
        }

        public OnlineTransducerModelConfig Transducer;
        public OnlineParaformerModelConfig Paraformer;
        public OnlineZipformer2CtcModelConfig Zipformer2Ctc;

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

        [MarshalAs(UnmanagedType.LPStr)]
        public string ModelingUnit;

        [MarshalAs(UnmanagedType.LPStr)]
        public string BpeVocab;
    }

}