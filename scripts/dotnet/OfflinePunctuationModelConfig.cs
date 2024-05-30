/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflinePunctuationModelConfig
    {
        public OfflinePunctuationModelConfig()
        {
            CtTransformer = "";
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        [MarshalAs(UnmanagedType.LPStr)]
        public string CtTransformer;

        public int NumThreads;

        public int Debug;

        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}
