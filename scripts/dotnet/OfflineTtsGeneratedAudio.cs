/// Copyright (c)  2024.5 by 东风破
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    public class OfflineTtsGeneratedAudio
    {
        public OfflineTtsGeneratedAudio(IntPtr p)
        {
            _handle = new HandleRef(this, p);
        }

        public bool SaveToWaveFile(String filename)
        {
            // A NULL handle means generation failed (the C API reports errors by
            // returning no audio). Report empty audio instead of dereferencing
            // IntPtr.Zero below.
            if (Handle == IntPtr.Zero) return false;
            Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
            byte[] utf8Filename = Encoding.UTF8.GetBytes(filename);
            byte[] utf8FilenameWithNull = new byte[utf8Filename.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Filename, utf8FilenameWithNull, utf8Filename.Length);
            utf8FilenameWithNull[utf8Filename.Length] = 0; // Null terminator
            int status = SherpaOnnxWriteWave(impl.Samples, impl.NumSamples, impl.SampleRate, utf8FilenameWithNull);
            return status == 1;
        }

        ~OfflineTtsGeneratedAudio()
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
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(Handle);
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
                if (Handle == IntPtr.Zero) return 0;
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.NumSamples;
            }
        }

        public int SampleRate
        {
            get
            {
                if (Handle == IntPtr.Zero) return 0;
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.SampleRate;
            }
        }

        public float[] Samples
        {
            get
            {
                if (Handle == IntPtr.Zero) return new float[0];
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));

                float[] samples = new float[impl.NumSamples];
                Marshal.Copy(impl.Samples, samples, 0, impl.NumSamples);
                return samples;
            }
        }

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineTtsGeneratedAudio(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxWriteWave(IntPtr samples, int n, int sample_rate, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Filename);
    }
}
