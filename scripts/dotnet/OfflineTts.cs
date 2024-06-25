/// Copyright (c)  2024.5 by 东风破
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    // IntPtr is actually a `const float*` from C++
    public delegate int OfflineTtsCallback(IntPtr samples, int n);

    public class OfflineTts : IDisposable
    {
        public OfflineTts(OfflineTtsConfig config)
        {
            IntPtr h = SherpaOnnxCreateOfflineTts(ref config);
            _handle = new HandleRef(this, h);
        }

        public OfflineTtsGeneratedAudio Generate(String text, float speed, int speakerId)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            IntPtr p = SherpaOnnxOfflineTtsGenerate(_handle.Handle, utf8Bytes, speakerId, speed);
            return new OfflineTtsGeneratedAudio(p);
        }

        public OfflineTtsGeneratedAudio GenerateWithCallback(String text, float speed, int speakerId, OfflineTtsCallback callback)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            IntPtr p = SherpaOnnxOfflineTtsGenerateWithCallback(_handle.Handle, utf8Bytes, speakerId, speed, callback);
            return new OfflineTtsGeneratedAudio(p);
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~OfflineTts()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineTts(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        public int SampleRate
        {
            get
            {
                return SherpaOnnxOfflineTtsSampleRate(_handle.Handle);
            }
        }

        public int NumSpeakers
        {
            get
            {
                return SherpaOnnxOfflineTtsNumSpeakers(_handle.Handle);
            }
        }

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineTts(ref OfflineTtsConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineTts(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineTtsSampleRate(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern int SherpaOnnxOfflineTtsNumSpeakers(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxOfflineTtsGenerate(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Text, int sid, float speed);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxOfflineTtsGenerateWithCallback(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Text, int sid, float speed, OfflineTtsCallback callback);
    }
}
