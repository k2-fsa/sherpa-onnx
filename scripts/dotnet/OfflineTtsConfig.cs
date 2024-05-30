/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineTtsConfig
    {
        public OfflineTtsConfig()
        {
            Model = new OfflineTtsModelConfig();
            RuleFsts = "";
            MaxNumSentences = 1;
            RuleFars = "";
        }
        public OfflineTtsModelConfig Model;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFsts;

        public int MaxNumSentences;

        [MarshalAs(UnmanagedType.LPStr)]
        public string RuleFars;
    }
}
