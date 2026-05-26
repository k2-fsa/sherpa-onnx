/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class DenoisedAudio
    {
        public DenoisedAudio(IntPtr p)
        {
            _handle = new HandleRef(this, p);
        }

        public bool SaveToWaveFile(String filename)
        {
            if (Handle == IntPtr.Zero)
            {
                return false;
            }

            Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
            return WriteWave(impl.Samples, impl.NumSamples, impl.SampleRate, filename);
        }

        public static bool SaveToWaveFile(float[] samples, int sampleRate, String filename)
        {
            if (samples == null || samples.Length == 0)
            {
                return false;
            }

            IntPtr p = Marshal.AllocHGlobal(samples.Length * sizeof(float));
            Marshal.Copy(samples, 0, p, samples.Length);
            bool ok = WriteWave(p, samples.Length, sampleRate, filename);
            Marshal.FreeHGlobal(p);
            return ok;
        }

        private static bool WriteWave(IntPtr samples, int numSamples, int sampleRate, String filename)
        {
            byte[] utf8Filename = Encoding.UTF8.GetBytes(filename);
            byte[] utf8FilenameWithNull = new byte[utf8Filename.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Filename, utf8FilenameWithNull, utf8Filename.Length);
            utf8FilenameWithNull[utf8Filename.Length] = 0; // Null terminator
            int status = SherpaOnnxWriteWave(samples, numSamples, sampleRate, utf8FilenameWithNull);
            return status == 1;
        }

        ~DenoisedAudio()
        {
            Cleanup();
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        private void Cleanup()
        {
            if (Handle != IntPtr.Zero)
            {
                SherpaOnnxDestroyDenoisedAudio(Handle);
            }

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        [StructLayout(LayoutKind.Sequential)]
        struct Impl
        {
            public IntPtr Samples;
            public int NumSamples;
            public int SampleRate;
        }

        private HandleRef _handle;
        public IntPtr Handle => _handle.Handle;

        public int NumSamples
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    return 0;
                }

                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.NumSamples;
            }
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

        public float[] Samples
        {
            get
            {
                if (Handle == IntPtr.Zero)
                {
                    return new float[0];
                }

                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));

                float[] samples = new float[impl.NumSamples];
                if (impl.NumSamples > 0 && impl.Samples != IntPtr.Zero)
                {
                    Marshal.Copy(impl.Samples, samples, 0, impl.NumSamples);
                }
                return samples;
            }
        }

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyDenoisedAudio(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxWriteWave(IntPtr samples, int n, int sample_rate, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Filename);
    }
}
