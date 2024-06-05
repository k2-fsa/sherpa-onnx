/// Copyright (c)  2024.5 by 东风破

using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System;

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
            Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
            int status = SherpaOnnxWriteWave(impl.Samples, impl.NumSamples, impl.SampleRate, filename);
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
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(Handle);

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
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.NumSamples;
            }
        }

        public int SampleRate
        {
            get
            {
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));
                return impl.SampleRate;
            }
        }

        public float[] Samples
        {
            get
            {
                Impl impl = (Impl)Marshal.PtrToStructure(Handle, typeof(Impl));

                float[] samples = new float[impl.NumSamples];
                Marshal.Copy(impl.Samples, samples, 0, impl.NumSamples);
                return samples;
            }
        }

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineTtsGeneratedAudio(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxWriteWave(IntPtr samples, int n, int sample_rate, [MarshalAs(UnmanagedType.LPStr)] string filename);
    }
}