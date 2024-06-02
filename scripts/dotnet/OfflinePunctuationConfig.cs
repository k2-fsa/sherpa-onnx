/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflinePunctuationConfig
    {
        public OfflinePunctuationConfig()
        {
            Model = new OfflinePunctuationModelConfig();
        }
        public OfflinePunctuationModelConfig Model;
    }
}

