/// Copyright (c)  2026  Xiaomi Corporation

using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    /// <summary>
    /// VoxLock target-speaker voice isolation model configuration.
    /// VoxLock keeps the voice of one speaker (the caller) and removes the
    /// other voices. The model is from https://github.com/arbazsiddiqui/voxlock
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OfflineSpeechDenoiserVoxLockModelConfig
    {
        public OfflineSpeechDenoiserVoxLockModelConfig()
        {
            EnrollModel = "";
            ChunkModel = "";
            VoiceSample = "";
        }

        /// <summary>
        /// Path to the VoxLock enroll model, e.g., enroll.onnx. It runs once on
        /// a voice sample of the speaker to keep.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string EnrollModel;

        /// <summary>
        /// Path to the VoxLock chunk model, e.g., chunk.onnx or
        /// chunk_int8.onnx. It runs on each chunk of the audio.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string ChunkModel;

        /// <summary>
        /// Optional. Path to a wave file with at least 3 seconds of the
        /// caller's voice. VoxLock keeps this voice and removes the others.
        /// From a longer file, it uses the loudest 3 seconds within the first
        /// 30 seconds. A file that is shorter or cannot be read is ignored, as
        /// if it were empty. If empty or ignored, the voice sample is taken
        /// from the input audio: the offline denoiser uses the loudest 3
        /// seconds in the first 15 seconds of the input, and the online
        /// denoiser passes the audio through unchanged until 3 seconds of
        /// speech have been heard, then uses that speech.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        public string VoiceSample;
    }
}
