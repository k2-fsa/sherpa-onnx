/// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class SourceSeparationOutput : IDisposable
    {
        public SourceSeparationOutput(IntPtr p)
        {
            _handle = new HandleRef(this, p);
        }

        public int SampleRate
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    return 0;
                }

                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.SampleRate;
            }
        }

        public int NumStems
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    return 0;
                }

                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.NumStems;
            }
        }

        // Return stem samples as float[][] where result[channel] is a float array
        public float[][] GetStemSamples(int stemIndex)
        {
            if (Handle == IntPtr.Zero)
            {
                return new float[0][];
            }

            Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
            if (stemIndex < 0 || stemIndex >= impl.NumStems)
            {
                return new float[0][];
            }

            int stemSize = Marshal.SizeOf(typeof(StemImpl));
            IntPtr stemPtr = new IntPtr(impl.Stems.ToInt64() + stemIndex * stemSize);
            StemImpl stem = (StemImpl)Marshal.PtrToStructure(stemPtr, typeof(StemImpl));

            int nc = stem.NumChannels;
            int n = stem.N;

            float[][] result = new float[nc][];
            IntPtr[] channelPtrs = new IntPtr[nc];
            Marshal.Copy(stem.Samples, channelPtrs, 0, nc);

            for (int c = 0; c < nc; ++c)
            {
                result[c] = new float[n];
                Marshal.Copy(channelPtrs[c], result[c], 0, n);
            }

            return result;
        }

        public bool SaveStemToWaveFile(int stemIndex, string filename)
        {
            if (Handle == IntPtr.Zero)
            {
                return false;
            }

            Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
            if (stemIndex < 0 || stemIndex >= impl.NumStems)
            {
                return false;
            }

            int stemSize = Marshal.SizeOf(typeof(StemImpl));
            IntPtr stemPtr = new IntPtr(impl.Stems.ToInt64() + stemIndex * stemSize);
            StemImpl stem = (StemImpl)Marshal.PtrToStructure(stemPtr, typeof(StemImpl));

            byte[] utf8Filename = Encoding.UTF8.GetBytes(filename);
            byte[] utf8FilenameWithNull = new byte[utf8Filename.Length + 1];
            Array.Copy(utf8Filename, utf8FilenameWithNull, utf8Filename.Length);
            utf8FilenameWithNull[utf8Filename.Length] = 0;

            int status = SherpaOnnxWriteWaveMultiChannel(stem.Samples, stem.N, impl.SampleRate, stem.NumChannels, utf8FilenameWithNull);
            return status == 1;
        }

        ~SourceSeparationOutput()
        {
            Cleanup();
        }

        public void Dispose()
        {
            Cleanup();
            System.GC.SuppressFinalize(this);
        }

        private void Cleanup()
        {
            if (Handle != IntPtr.Zero)
            {
                SherpaOnnxDestroySourceSeparationOutput(Handle);
            }

            _handle = new HandleRef(this, IntPtr.Zero);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct StemImpl
        {
            public IntPtr Samples; // float**
            public int NumChannels;
            public int N;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Stems; // const SherpaOnnxSourceSeparationStem*
            public int NumStems;
            public int SampleRate;
        }

        private HandleRef _handle;
        public IntPtr Handle => _handle.Handle;

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroySourceSeparationOutput(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxWriteWaveMultiChannel(IntPtr samples, int n, int sample_rate, int num_channels, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Filename);
    }
}
