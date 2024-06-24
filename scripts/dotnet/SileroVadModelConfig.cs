/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SileroVadModelConfig
    {
        public SileroVadModelConfig()
        {
            Model = "";
            Threshold = 0.5F;
            MinSilenceDuration = 0.5F;
            MinSpeechDuration = 0.25F;
            WindowSize = 512;
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        public float Threshold;

        public float MinSilenceDuration;

        public float MinSpeechDuration;

        public int WindowSize;
    }
}
