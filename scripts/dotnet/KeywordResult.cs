/// Copyright (c)  2024  Xiaomi Corporation

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class KeywordResult
    {
        public KeywordResult(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _keyword = Marshal.PtrToStringUTF8(impl.Keyword);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Keyword;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Keyword, stringBuffer, 0, length);
            _keyword = Encoding.UTF8.GetString(stringBuffer);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Keyword;
        }

        private String _keyword;
        public String Keyword => _keyword;
    }
}
