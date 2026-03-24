/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class OnlinePunctuation : IDisposable
    {
        public OnlinePunctuation(OnlinePunctuationConfig config)
        {
            IntPtr h = SherpaOnnxCreateOnlinePunctuation(ref config);
            _handle = new HandleRef(this, h);
        }

        public String AddPunct(String text)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            byte[] utf8BytesWithNull = new byte[utf8Bytes.Length + 1];
            Array.Copy(utf8Bytes, utf8BytesWithNull, utf8Bytes.Length);
            utf8BytesWithNull[utf8Bytes.Length] = 0;

            IntPtr p = SherpaOnnxOnlinePunctuationAddPunct(
                _handle.Handle, utf8BytesWithNull);

            string s = "";
            int length = 0;

            unsafe
            {
                byte* b = (byte*)p;
                if (b != null)
                {
                    while (*b != 0)
                    {
                        ++b;
                        length += 1;
                    }
                }
            }

            if (length > 0)
            {
                byte[] stringBuffer = new byte[length];
                Marshal.Copy(p, stringBuffer, 0, length);
                s = Encoding.UTF8.GetString(stringBuffer);
            }

            SherpaOnnxOnlinePunctuationFreeText(p);
            return s;
        }

        public void Dispose()
        {
            Cleanup();
            System.GC.SuppressFinalize(this);
        }

        ~OnlinePunctuation()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOnlinePunctuation(_handle.Handle);
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOnlinePunctuation(
            ref OnlinePunctuationConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOnlinePunctuation(
            IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOnlinePunctuationAddPunct(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)]
            byte[] utf8Text);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxOnlinePunctuationFreeText(
            IntPtr p);
    }
}
