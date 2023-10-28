using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TTS.Struct
{
    /// <summary>
    /// 生成语音结果
    /// </summary> 
    public sealed partial class SherpaOnnxGeneratedAudioResult : IDisposable
    {
        public const string Filename = "sherpa-onnx-c-api";

        /// <summary>
        /// 销毁非托管内存
        /// </summary>
        /// <param name="ttsGenerateIntptr"></param>
        [DllImport(Filename)]
        private static extern void SherpaOnnxDestroyOfflineTtsGeneratedAudio(IntPtr ttsGenerateIntptr);

        [DllImport(Filename)]
        private static extern int SherpaOnnxWriteWave(IntPtr q, int n, int sample_rate, string filename);

        /// <summary>
        /// 音频数据比特
        /// </summary>
        public const int AudioDataBit = 16;
        /// <summary>
        /// 单通道
        /// </summary>
        public const int Channels = 1;

        /// <summary>
        /// 原生句柄
        /// </summary>
        internal IntPtr thisHandle;

        internal readonly IntPtr audioData;
        internal readonly int dataSize;

        /// <summary>
        /// 采样率
        /// </summary>
        public readonly int sample_rate;

        /// <summary>
        /// 音频数据指针
        /// </summary>
        public IntPtr AudioDataIntPtr => audioData;

        /// <summary>
        /// 数据的大小
        /// </summary>
        public unsafe int AudioDataLength
        {
            get
            {
                return dataSize;

                //float* buffer = (float*)audioData;
                //while (*buffer != 0)
                //    ++buffer;
                //return (int)(buffer - (float*)audioData);
            }
        }

        /// <summary>
        /// 获得音频数据 float[]
        /// 这个内部创建一个数组
        /// </summary>
        public unsafe float[] AudioFloatData
        {
            get
            {
                int length = AudioDataLength;

                float[] floatAudioData = new float[length];
                Marshal.Copy(audioData, floatAudioData, 0, floatAudioData.Length);
                return floatAudioData;
            }
        }


        /// <summary>
        /// 获得音频数据 byte[]
        /// 这个内部创建一个数组
        /// </summary>
        public byte[] AudioByteData
        {
            get
            {
                byte[] bytes = new byte[AudioDataLength * 2];
                ReadData(bytes, 0);
                return bytes;
            }
        }

        internal SherpaOnnxGeneratedAudioResult(IntPtr intPtr, SherpaOnnxGeneratedAudio sherpaOnnx)
        {
            this.thisHandle = intPtr;
            this.audioData = sherpaOnnx.audioData;
            this.dataSize = sherpaOnnx.dataSize;
            this.sample_rate = sherpaOnnx.sample_rate;
        }

        ~SherpaOnnxGeneratedAudioResult()
        {
            Dispose();
        }

        /// <summary>
        /// 读取数据
        /// 没有垃圾产生，自己传递数组进来
        /// </summary>
        /// <param name="audioFloats">数组</param>
        /// <param name="offset">数组那个位置写入</param>
        /// <returns>写入了多少个</returns>
        public int ReadData(float[] audioFloats, int offset)
        {
            int length = AudioDataLength;

            int c = audioFloats.Length - offset;
            length = c >= length ? length : c;

            Marshal.Copy(audioData, audioFloats, offset, length);
            return length;
        }

        /// <summary>
        /// 读取数据
        /// 这个内部转换成byte[] 音频数组
        /// 没有垃圾产生，自己传递数组进来
        /// </summary>
        /// <param name="audioFloats">数组，这个长度需要是AudioDataLength*2大小</param>
        /// <param name="offset">数组那个位置写入</param>
        /// <returns>写入了多少个</returns>
        public int ReadData(byte[] audioFloats, int offset)
        {
            //因为是16bit存储音频数据，所以float会转换成两个字节存储
            var audiodata = AudioFloatData;

            int length = audiodata.Length * 2;

            int c = audioFloats.Length - offset;
            c = c % 2 == 0 ? c : c - 1;

            length = c >= length ? length : c;

            int p = length / 2;

            for (int i = 0; i < p; i++)
            {
                short value = (short)(audiodata[i] * short.MaxValue);

                audioFloats[offset++] = (byte)value;
                audioFloats[offset++] = (byte)(value >> 8);
            }

            return length;

        }

        /// <summary>
        /// 写入WAV音频数据
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public bool WriteWAVFile(string filename)
        {
            return 1 == SherpaOnnxWriteWave(audioData, this.dataSize, this.sample_rate, filename);
        }

        public void Dispose()
        {
            if (this.thisHandle != IntPtr.Zero)
            {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(this.thisHandle);
                GC.SuppressFinalize(this);
                this.thisHandle = IntPtr.Zero;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct SherpaOnnxGeneratedAudio
    {
        internal readonly IntPtr audioData;
        internal readonly int dataSize;

        /// <summary>
        /// 采样率
        /// </summary>
        public readonly int sample_rate;
    }
}
