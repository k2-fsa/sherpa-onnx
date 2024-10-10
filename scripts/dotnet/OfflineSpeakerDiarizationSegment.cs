/// Copyright (c)  2024  Xiaomi Corporation
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{

    public class OfflineSpeakerDiarizationSegment
    {
        public OfflineSpeakerDiarizationSegment(IntPtr handle)
        {
          Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

          Start = impl.Start;
          End = impl.End;
          Speaker = impl.Speaker;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public float Start;
            public float End;
            public int Speaker;
        }

        public float Start;
        public float End;
        public int Speaker;
    }
}

