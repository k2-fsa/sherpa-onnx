/// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

namespace SherpaOnnx
{
    public class SpeechSegment
    {
      public SpeechSegment(IntPtr handle)
      {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            _start = impl.Start;

            unsafe
            {
                float* t = (float*)impl.Samples;
                _samples = new float[impl.Count];
                fixed (float* pTarget = _samples)
                {
                    for (int i = 0; i < impl.Count; i++)
                    {
                        pTarget[i] = t[i];
                    }
                }
            }
      }

      public int _start;
      public int Start => _start;

      private float[] _samples;
      public float[] Samples => _samples;

      [StructLayout(LayoutKind.Sequential)]
      struct Impl
      {
          public int Start;
          public IntPtr Samples;
          public int Count;
      }
    }
}
