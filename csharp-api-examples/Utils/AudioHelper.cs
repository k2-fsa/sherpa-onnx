using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// audio processing
/// Copyright (c)  2023 by manyeyes
/// </summary>
public class AudioHelper
{
    public static float[] GetFileSamples(string wavFilePath, ref TimeSpan duration)
    {
        if (!File.Exists(wavFilePath))
        {
            Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
            return new float[1];
        }
        AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
        byte[] datas = new byte[_audioFileReader.Length];
        _audioFileReader.Read(datas, 0, datas.Length);
        duration = _audioFileReader.TotalTime;
        float[] wavdata = new float[datas.Length / sizeof(float)];
        Buffer.BlockCopy(datas, 0, wavdata, 0, datas.Length);
        return wavdata;
    }

    public static List<float[]> GetChunkSamplesList(string wavFilePath, ref TimeSpan duration)
    {
        List<float[]> wavdatas = new List<float[]>();
        if (!File.Exists(wavFilePath))
        {
            Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
            wavdatas.Add(new float[1]);
            return wavdatas;
        }
        AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
        byte[] datas = new byte[_audioFileReader.Length];
        int chunkSize = 16000;// datas.Length / sizeof(float);
        int chunkNum = (int)Math.Ceiling((double)datas.Length / chunkSize);
        for (int i = 0; i < chunkNum; i++)
        {
            int offset = 0;
            int dataCount = 0;
            if (Math.Abs(datas.Length - i * chunkSize) > chunkSize)
            {
                offset = i * chunkSize;
                dataCount = chunkSize;
            }
            else
            {
                offset = i * chunkSize;
                dataCount = datas.Length - i * chunkSize;
            }
            _audioFileReader.Read(datas, offset, dataCount);
            duration += _audioFileReader.TotalTime;
            float[] wavdata = new float[chunkSize / sizeof(float)];
            Buffer.BlockCopy(datas, offset, wavdata, 0, dataCount);
            wavdatas.Add(wavdata);

        }
        return wavdatas;
    }
}
