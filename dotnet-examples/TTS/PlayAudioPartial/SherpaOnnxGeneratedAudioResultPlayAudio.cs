using NAudio.Wave;

namespace TTS.Struct
{
    public sealed partial class SherpaOnnxGeneratedAudioResult
    {
        private WaveOutEvent waveOut;
        private WaveFormat waveFormat;
        private BufferedWaveProvider bufferedWaveProvider;

        private int bufferLength = 1;

        public TimeSpan? AudioDuration => bufferedWaveProvider?.BufferedDuration;

        public float PlayProgress => (waveOut?.GetPosition() * 1.0f / bufferLength).Value;

        public void Play()
        {
            waveOut ??= new WaveOutEvent();

            waveFormat ??= new WaveFormat(sample_rate, AudioDataBit, Channels); // 32-bit 浮点，单声道

            if (bufferedWaveProvider == null)
            {
                bufferedWaveProvider ??= new BufferedWaveProvider(waveFormat);

                var buffer = AudioByteData;

                bufferLength = buffer.Length;

                bufferedWaveProvider.AddSamples(buffer, 0, bufferLength);
                bufferedWaveProvider.BufferLength = bufferLength;
                waveOut.Init(bufferedWaveProvider);
            }
            waveOut.Play();
        }

        public void Stop()
        {
            waveOut?.Stop();
        }

    }
}
