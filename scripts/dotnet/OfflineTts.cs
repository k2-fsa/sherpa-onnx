/// Copyright (c)  2024.5 by 东风破
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SherpaOnnx
{
    // IntPtr is actually a `const float*` from C++
    public delegate int OfflineTtsCallback(IntPtr samples, int n);
    public delegate int OfflineTtsCallbackProgress(IntPtr samples, int n, float progress);

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
            byte[] utf8BytesWithNull = new byte[utf8Bytes.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Bytes, utf8BytesWithNull, utf8Bytes.Length);
            utf8BytesWithNull[utf8Bytes.Length] = 0; // Null terminator
            IntPtr p = SherpaOnnxOfflineTtsGenerate(_handle.Handle, utf8BytesWithNull, speakerId, speed);
            return new OfflineTtsGeneratedAudio(p);
        }

        public OfflineTtsGeneratedAudio GenerateWithCallback(String text, float speed, int speakerId, OfflineTtsCallback callback)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            byte[] utf8BytesWithNull = new byte[utf8Bytes.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Bytes, utf8BytesWithNull, utf8Bytes.Length);
            utf8BytesWithNull[utf8Bytes.Length] = 0; // Null terminator
            IntPtr p = SherpaOnnxOfflineTtsGenerateWithCallback(_handle.Handle, utf8BytesWithNull, speakerId, speed, callback);
            return new OfflineTtsGeneratedAudio(p);
        }

        public OfflineTtsGeneratedAudio GenerateWithCallbackProgress(String text, float speed, int speakerId, OfflineTtsCallbackProgress callback)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
            byte[] utf8BytesWithNull = new byte[utf8Bytes.Length + 1]; // +1 for null terminator
            Array.Copy(utf8Bytes, utf8BytesWithNull, utf8Bytes.Length);
            utf8BytesWithNull[utf8Bytes.Length] = 0; // Null terminator
            IntPtr p = SherpaOnnxOfflineTtsGenerateWithProgressCallback(_handle.Handle, utf8BytesWithNull, speakerId, speed, callback);
            return new OfflineTtsGeneratedAudio(p);
        }

        /// <summary>
        /// Generate speech with ZipVoice zero-shot cloning.
        /// </summary>
        /// <param name="text">Synthesis text. Must not be null or whitespace.</param>
        /// <param name="promptText">Transcription of <paramref name="promptSamples"/> (can be empty).</param>
        /// <param name="promptSamples">PCM samples in [-1, 1] of the prompt voice (can be null or empty).</param>
        /// <param name="promptSampleRate">Sample rate of <paramref name="promptSamples"/>. Must be &gt; 0 when samples are provided.</param>
        /// <param name="speed">Speaking speed multiplier. Must be a positive finite value.</param>
        /// <param name="numSteps">ZipVoice inference steps. Must be &gt; 0.</param>
        /// <returns>
        /// A handle to generated audio, or <c>null</c> if native generation failed.
        /// The caller is responsible for disposing the returned <see cref="OfflineTtsGeneratedAudio"/>.
        /// </returns>
        /// <exception cref="ObjectDisposedException">Thrown if the TTS instance was disposed.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="text"/> is null/whitespace.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if numeric arguments are invalid.</exception>
        public OfflineTtsGeneratedAudio GenerateWithZipvoice(
            string text, string promptText, float[] promptSamples,
            int promptSampleRate, float speed, int numSteps)
        {
            if (_handle.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(OfflineTts));

            if (text == null || text.Trim().Length == 0)
                throw new ArgumentException("Text must not be null or whitespace.", "text");

            if (float.IsNaN(speed) || float.IsInfinity(speed) || speed <= 0f)
                throw new ArgumentOutOfRangeException("speed", "Speed must be a positive finite value.");

            if (numSteps <= 0)
                throw new ArgumentOutOfRangeException(nameof(numSteps), "numSteps must be > 0.");

            if (promptSamples != null && promptSamples.Length > 0 && promptSampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(promptSampleRate), "Must be > 0 when promptSamples are provided.");

            // Helper: UTF-8 bytes with trailing NUL (C-string)
            static byte[] ToUtf8Z(string s)
            {
                if (string.IsNullOrEmpty(s))
                    return new byte[1]; // just '\0'
                var bytes = Encoding.UTF8.GetBytes(s);
                var z = new byte[bytes.Length + 1];
                Array.Copy(bytes, z, bytes.Length);
                // last byte remains 0
                return z;
            }

            var utf8Text = ToUtf8Z(text);
            var utf8PromptText = ToUtf8Z(promptText);

            IntPtr pSamples = IntPtr.Zero;
            int n = 0;

            try
            {
                if (promptSamples != null && promptSamples.Length > 0)
                {
                    n = promptSamples.Length;
                    int byteSize = checked(n * sizeof(float));
                    pSamples = Marshal.AllocHGlobal(byteSize);
                    Marshal.Copy(promptSamples, 0, pSamples, n);
                }

                IntPtr p = SherpaOnnxOfflineTtsGenerateWithZipvoice(
                    _handle.Handle,
                    utf8Text,
                    utf8PromptText,
                    pSamples,
                    n,
                    promptSampleRate,
                    speed,
                    numSteps);

                return p == IntPtr.Zero ? null : new OfflineTtsGeneratedAudio(p);
            }
            finally
            {
                if (pSamples != IntPtr.Zero)
                    Marshal.FreeHGlobal(pSamples);
            }
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

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxOfflineTtsGenerateWithProgressCallback(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Text, int sid, float speed, OfflineTtsCallbackProgress callback);

        [DllImport(Dll.Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxOfflineTtsGenerateWithZipvoice(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Text, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8PromptText, IntPtr promptSamples, int nPrompt, int promptSampleRate, float speed, int numSteps);
    }
}
