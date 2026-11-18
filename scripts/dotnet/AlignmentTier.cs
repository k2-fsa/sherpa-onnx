/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class AlignmentTier
    {
        public AlignmentTier(IntPtr handle)
        {
            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _name = Marshal.PtrToStringUTF8(impl.Name);

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

            _intervals = AlignmentInterval.FromArray(impl.Intervals, impl.NumIntervals);
        }

        // Unmarshal a C array of SherpaOnnxForcedAlignmentTier.
        internal static AlignmentTier[] FromArray(IntPtr p, int n)
        {
            int size = Marshal.SizeOf(typeof(Impl));
            AlignmentTier[] ans = new AlignmentTier[n];
            for (int i = 0; i < n; ++i)
            {
                IntPtr q = new IntPtr(p.ToInt64() + i * size);
                ans[i] = new AlignmentTier(q);
            }
            return ans;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Name;
            public int NumIntervals;
            public IntPtr Intervals;
        }

        private string _name;

        /// <summary>Tier name: "phones" or "words"; never empty.</summary>
        public string Name => _name;

        private AlignmentInterval[] _intervals;

        /// <summary>
        /// The intervals of this tier, in time order. They tile the whole
        /// utterance: silence is represented by intervals with an empty
        /// text. Tiers[0] of a result is the "phones" tier, Tiers[1] the
        /// "words" tier.
        /// </summary>
        public AlignmentInterval[] Intervals => _intervals;
    }
}
