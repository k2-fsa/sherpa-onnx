/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

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
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Text;
        }

        private String _text;
        public String Text => _text;
    }


}