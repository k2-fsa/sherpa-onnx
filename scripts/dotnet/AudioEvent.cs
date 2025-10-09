/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{

    public class AudioEvent
    {
        public AudioEvent(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _text = Marshal.PtrToStringUTF8(impl.Text);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Name;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Name, stringBuffer, 0, length);
            _name = Encoding.UTF8.GetString(stringBuffer);

            _index = impl.Index;
            _prob = impl.Prob;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Name;
            public int Index;
            public float Prob;
        }

        private String _name;
        public String Name => _name;

        private int _index;
        public int Index => _index;

        private float _prob;
        public float Prob => _prob;
    }
}
