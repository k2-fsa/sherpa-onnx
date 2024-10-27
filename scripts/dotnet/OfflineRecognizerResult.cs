/// Copyright (c)  2024.5 by 东风破

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{

    public class OfflineRecognizerResult
    {
        public OfflineRecognizerResult(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _text = Marshal.PtrToStringUTF8(impl.Text);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Text;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Text, stringBuffer, 0, length);
            _text = Encoding.UTF8.GetString(stringBuffer);

            _tokens = new String[impl.Count];

            unsafe
            {
                byte* buf = (byte*)impl.Tokens;
                for (int i = 0; i < impl.Count; i++)
                {
                    length = 0;
                    byte* start = buf;
                    while (*buf != 0)
                    {
                        ++buf;
                        length += 1;
                    }
                    ++buf;

                    stringBuffer = new byte[length];
                    fixed (byte* pTarget = stringBuffer)
                    {
                        for (int k = 0; k < length; k++)
                        {
                            pTarget[k] = start[k];
                        }
                    }

                    _tokens[i] = Encoding.UTF8.GetString(stringBuffer);
                }
            }

            unsafe
            {
              if (impl.Timestamps != IntPtr.Zero)
              {
                float *t = (float*)impl.Timestamps;
                _timestamps = new float[impl.Count];
                fixed (float* f = _timestamps)
                {
                  for (int k = 0; k < impl.Count; k++)
                  {
                    f[k] = t[k];
                  }
                }
              }
            }

        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Text;
            public IntPtr Timestamps;
            public int Count;
            public IntPtr Tokens;
        }

        private String _text;
        public String Text => _text;

        private String[] _tokens;
        public String[] Tokens => _tokens;

        private float[] _timestamps;
        public float[] Timestamps => _timestamps;
    }
}
