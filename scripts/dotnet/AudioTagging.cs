/// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.Runtime.InteropServices;
using System.Text;


namespace SherpaOnnx
{
    public class AudioTagging : IDisposable
    {
        public AudioTagging(AudioTaggingConfig config)
        {
            IntPtr h = SherpaOnnxCreateAudioTagging(ref config);
            _handle = new HandleRef(this, h);
        }

        public OfflineStream CreateStream()
        {
            IntPtr p = SherpaOnnxAudioTaggingCreateOfflineStream(_handle.Handle);
            return new OfflineStream(p);
        }

        // if top_k <= 0, then conig.TopK is used
        // if top_k > 0, then confing.TopK is ignored
        public AudioEvent[] Compute(OfflineStream stream, int top_k)
        {
            IntPtr p = SherpaOnnxAudioTaggingCompute(_handle.Handle, stream.Handle, top_k);

            var result = new List<AudioEvent>();

            if (p == IntPtr.Zero)
            {
              return result.ToArray();
            }

            int index = 0;
            while (true)
            {
              IntPtr e = Marshal.ReadIntPtr(p, index * IntPtr.Size);
              if (e = IntPtr.Zero)
              {
                break;
              }

              AudioEvent ae = new AudioEvent(e);
              result.Add(ae);
            }

            SherpaOnnxAudioTaggingFreeResults(p);

            return result.ToArray();
        }

        public void Dispose()
        {
            Cleanup();
            // Prevent the object from being placed on the
            // finalization queue
            System.GC.SuppressFinalize(this);
        }

        ~AudioTagging()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            SherpaOnnxDestroyAudioTagging(_handle.Handle);

            // Don't permit the handle to be used again.
            _handle = new HandleRef(this, IntPtr.Zero);
        }

        private HandleRef _handle;


        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxCreateAudioTagging(ref AudioTaggingConfig config);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxDestroyAudioTagging(IntPtr handle);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOfflinePunctuationAddPunct(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Text);

        [DllImport(Dll.Filename)]
        private static extern IntPtr SherpaOnnxAudioTaggingCompute(IntPtr handle, IntPtr stream, int top_k);

        [DllImport(Dll.Filename)]
        private static extern void SherpaOnnxAudioTaggingFreeResults(IntPtr p);
    }
}

