/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
/// Copyright (c)  2023 by manyeyes
/// Copyright (c)  2024.5 by 东风破

using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{

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

}