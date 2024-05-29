/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{

    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsModelConfig
    {
        public OfflineTtsModelConfig()
        {
            Vits = new OfflineTtsVitsModelConfig();
            NumThreads = 1;
            Debug = 0;
            Provider = "cpu";
        }

        public OfflineTtsVitsModelConfig Vits;
        public int NumThreads;
        public int Debug;
        [MarshalAs(UnmanagedType.LPStr)]
        public string Provider;
    }
}