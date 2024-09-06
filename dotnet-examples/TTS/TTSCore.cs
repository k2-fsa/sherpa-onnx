using System.Runtime.InteropServices;
using TTS.Struct;

namespace TTS
{
    internal sealed class TTSCore : IDisposable
    {
        public const string Filename = "sherpa-onnx-c-api";

        [DllImport(Filename, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr SherpaOnnxCreateOfflineTts(SherpaOnnxOfflineTtsConfig handle);

        [DllImport(Filename)]
        private static extern IntPtr SherpaOnnxOfflineTtsGenerate(IntPtr createOfflineTtsIntptr, IntPtr text, int sid, float speed);

        [DllImport(Filename)]
        private static extern void SherpaOnnxDestroyOfflineTts(IntPtr intPtr);

        /// <summary>
        /// 原生句柄
        /// </summary>
        private IntPtr thisHandle;

        public TTSCore(SherpaOnnxOfflineTtsConfig modelConfig)
        {
          IntPtr ttsHandle = SherpaOnnxCreateOfflineTts(modelConfig);
          if (ttsHandle == IntPtr.Zero)
          {
            throw new InvalidOperationException("Failed to create SherpaOnnx TTS engine.");
          }
          thisHandle = ttsHandle;
        }

        /// <summary>
        /// 文字转语音
        /// </summary>
        /// <param name="text">文字</param>
        /// <param name="sid">音色</param>
        /// <param name="speed">速度</param>
        /// <returns></returns>
        public SherpaOnnxGeneratedAudioResult ToSpeech(string text, int sid, float speed = 1f)
        {
            var result = SherpaOnnxOfflineTtsGenerate(thisHandle, Marshal.StringToCoTaskMemUTF8(text), sid, speed);
            SherpaOnnxGeneratedAudio impl = (SherpaOnnxGeneratedAudio)Marshal.PtrToStructure(result, typeof(SherpaOnnxGeneratedAudio));
            return new SherpaOnnxGeneratedAudioResult(result, impl);
        }

        /// <summary>
        /// 文字转语音
        /// </summary>
        /// <param name="text">文字</param>
        /// <param name="sid">音色</param>
        /// <param name="speed">速度</param>
        /// <returns></returns>
        public Task<SherpaOnnxGeneratedAudioResult> ToSpeechAsync(string text, int sid, float speed = 1f)
        {
            return Task.Run(() => ToSpeech(text, sid, speed));
        }

        ~TTSCore()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (this.thisHandle != IntPtr.Zero)
            {
                SherpaOnnxDestroyOfflineTts(this.thisHandle);
                GC.SuppressFinalize(this);
                this.thisHandle = IntPtr.Zero;
            }
        }
    }
}
