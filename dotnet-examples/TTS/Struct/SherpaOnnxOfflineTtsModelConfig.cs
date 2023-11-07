using System.Runtime.InteropServices;

namespace TTS.Struct
{
    [StructLayout(LayoutKind.Sequential)]
    public struct SherpaOnnxOfflineTtsModelConfig
    {
        /// <summary>
        /// 模型配置
        /// </summary>
        public SherpaOnnxOfflineTtsVitsModelConfig vits;
        /// <summary>
        /// 线程数
        /// </summary>
        public int num_threads;
        public int debug;
        /// <summary>
        /// 使用cpu
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string provider;
    }
}
