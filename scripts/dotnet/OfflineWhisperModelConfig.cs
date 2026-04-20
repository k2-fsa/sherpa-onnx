/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
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
            EnableTokenTimestamps = 0;
            EnableSegmentTimestamps = 0;
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
        public int EnableTokenTimestamps;
        public int EnableSegmentTimestamps;
    }

}
