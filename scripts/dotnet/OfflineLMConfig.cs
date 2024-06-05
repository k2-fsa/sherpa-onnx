/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineLMConfig
    {
        public OfflineLMConfig()
        {
            Model = "";
            Scale = 0.5F;
        }
        [MarshalAs(UnmanagedType.LPStr)]
        public string Model;

        public float Scale;
    }
}
