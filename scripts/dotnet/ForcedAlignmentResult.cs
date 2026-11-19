/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    /// <summary>
    /// Result of ForcedAligner.Compute(): the phone and word boundaries of
    /// one utterance and the same content as a Praat TextGrid.
    ///
    /// Tiers[0] is the "phones" tier, Tiers[1] the "words" tier. The result
    /// is empty (no tiers) when the alignment failed.
    /// </summary>
    public class ForcedAlignmentResult
    {
        public ForcedAlignmentResult(IntPtr handle)
        {
            if (handle == IntPtr.Zero)
            {
                // empty result: the alignment failed
                _duration = 0;
                _textGrid = "";
                _tiers = new AlignmentTier[0];
                return;
            }

            Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));

            // PtrToStringUTF8() requires .net standard 2.1
            // _textGrid = Marshal.PtrToStringUTF8(impl.Textgrid);

            int length = 0;

            unsafe
            {
                byte* buffer = (byte*)impl.Textgrid;
                while (*buffer != 0)
                {
                    ++buffer;
                    length += 1;
                }
            }

            byte[] stringBuffer = new byte[length];
            Marshal.Copy(impl.Textgrid, stringBuffer, 0, length);
            _textGrid = Encoding.UTF8.GetString(stringBuffer);

            _duration = impl.Duration;
            _tiers = AlignmentTier.FromArray(impl.Tiers, impl.NumTiers);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public float Duration;
            public IntPtr Textgrid;
            public int NumTiers;
            public IntPtr Tiers;
        }

        private float _duration;
        public float Duration => _duration;

        private string _textGrid;

        /// <summary>
        /// The whole alignment as a Praat TextGrid string; the same as
        /// ToTextGrid(). Save it to a .TextGrid file and open it with Praat
        /// (https://www.fon.hum.uva.nl/praat/) to visualize the phone and
        /// word boundaries.
        /// </summary>
        public string TextGrid => _textGrid;

        public string ToTextGrid()
        {
            return _textGrid;
        }

        private AlignmentTier[] _tiers;

        /// <summary>
        /// The alignment tiers. Tiers[0] is the "phones" tier and Tiers[1]
        /// the "words" tier. The result is empty (no tiers) when the
        /// alignment failed.
        /// </summary>
        public AlignmentTier[] Tiers => _tiers;
    }
}
