/// Copyright (c)  2024.5 by 东风破
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class SpokenLanguageIdentificationResult
    {
        public SpokenLanguageIdentificationResult(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _text = Marshal.PtrToStringUTF8(impl.Text);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Lang;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Lang, stringBuffer, 0, length);
            _lang = Encoding.UTF8.GetString(stringBuffer);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Lang;
        }

        private String _lang;
        public String Lang => _lang;
    }
}
