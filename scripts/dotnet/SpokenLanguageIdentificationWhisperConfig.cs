/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

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