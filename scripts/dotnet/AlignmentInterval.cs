/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class AlignmentInterval
    {
        public AlignmentInterval(IntPtr handle)
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

            _start = impl.Start;
            _end = impl.End;
        }

        // Unmarshal a C array of SherpaOnnxForcedAlignmentInterval.
        internal static AlignmentInterval[] FromArray(IntPtr p, int n)
        {
            int size = Marshal.SizeOf(typeof(Impl));
            AlignmentInterval[] ans = new AlignmentInterval[n];
            for (int i = 0; i < n; ++i)
            {
                IntPtr q = new IntPtr(p.ToInt64() + i * size);
                ans[i] = new AlignmentInterval(q);
            }
            return ans;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public float Start;
            public float End;
            public IntPtr Text;
        }

        private float _start;
        public float Start => _start;

        private float _end;
        public float End => _end;

        private string _text;

        /// <summary>
        /// The text of this interval: the word for a word interval, the
        /// phone symbol for a phone interval. It is empty for a silence gap
        /// in the "words" tier.
        /// </summary>
        public string Text => _text;
    }
}
