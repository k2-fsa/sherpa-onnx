/// Copyright (c)  2024  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeakerSegmentationModelConfig
    {
        public OfflineSpeakerSegmentationModelConfig()
        {
            Pyannote = new OfflineSpeakerSegmentationPyannoteModelConfig();
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineSpeakerSegmentationPyannoteModelConfig Pyannote;

        /// Number of threads used to run the neural network model
        public int NumThreads;

        /// true to print debug information of the model
        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}


