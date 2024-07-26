/// Copyright (c)  2024.5 by 东风破

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OfflineRecognizer : IDisposable
    {
        public OfflineRecognizer(OfflineRecognizerConfig config)
        {
            IntPtr h = SherpaOnnxCreateOfflineRecognizer(ref config);
            _handle = new HandleRef(this, h);
        }

        public OfflineStream CreateStream()
        {
            IntPtr p = SherpaOnnxCreateOfflineStream(_handle.Handle);
            return new OfflineStream(p);
        }

        public void Decode(OfflineStream stream)
        {
            Decode(_handle.Handle, stream.Handle);
        }

        // The caller should ensure all passed streams are ready for decoding.
        public void Decode(IEnumerable<OfflineStream> streams)
        {
            // TargetFramework=net20 does not support System.Linq
            // IntPtr[] ptrs = streams.Select(s => s.Handle).ToArray();
            List<IntPtr> list = new List<IntPtr>();
            foreach (OfflineStream s in streams)
            {
              list.Add(s.Handle);
            }
            IntPtr[] ptrs = list.ToArray();
            Decode(_handle.Handle, ptrs, ptrs.Length);
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~OfflineRecognizer()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyOfflineRecognizer(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineRecognizer(ref OfflineRecognizerConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyOfflineRecognizer(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateOfflineStream(IntPtr handle);

        [DllImport(Dll.Filename, EntryPoint = "SherpaOnnxDecodeOfflineStream")]
        private static extern void Decode(IntPtr handle, IntPtr stream);

        [DllImport(Dll.Filename, EntryPoint = "SherpaOnnxDecodeMultipleOfflineStreams")]
        private static extern void Decode(IntPtr handle, IntPtr[] streams, int n);
    }

}
