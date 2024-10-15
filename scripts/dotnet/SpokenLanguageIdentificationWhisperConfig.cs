/// Copyright (c)  2024.5 by 东风破

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
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

}