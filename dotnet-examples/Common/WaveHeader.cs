// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.IO;

using System.Runtime.InteropServices;

namespace SherpaOnnx
{

  [StructLayout(LayoutKind.Sequential)]
  public struct WaveHeader
  {
    public Int32 ChunkID;
    public Int32 ChunkSize;
    public Int32 Format;
    public Int32 SubChunk1ID;
    public Int32 SubChunk1Size;
    public Int16 AudioFormat;
    public Int16 NumChannels;
    public Int32 SampleRate;
    public Int32 ByteRate;
    public Int16 BlockAlign;
    public Int16 BitsPerSample;
    public Int32 SubChunk2ID;
    public Int32 SubChunk2Size;

    public bool Validate()
    {
      if (ChunkID != 0x46464952)
      {
        Console.WriteLine($"Invalid chunk ID: 0x{ChunkID:X}. Expect 0x46464952");
        return false;
      }

      //               E V A W
      if (Format != 0x45564157)
      {
        Console.WriteLine($"Invalid format: 0x{Format:X}. Expect 0x45564157");
        return false;
      }

      //                      t m f
      if (SubChunk1ID != 0x20746d66)
      {
        Console.WriteLine($"Invalid SubChunk1ID: 0x{SubChunk1ID:X}. Expect 0x20746d66");
        return false;
      }

      if (SubChunk1Size != 16)
      {
        Console.WriteLine($"Invalid SubChunk1Size: {SubChunk1Size}. Expect 16");
        return false;
      }

      if (AudioFormat != 1)
      {
        Console.WriteLine($"Invalid AudioFormat: {AudioFormat}. Expect 1");
        return false;
      }

      if (NumChannels != 1)
      {
        Console.WriteLine($"Invalid NumChannels: {NumChannels}. Expect 1");
        return false;
      }

      if (ByteRate != (SampleRate * NumChannels * BitsPerSample / 8))
      {
        Console.WriteLine($"Invalid byte rate: {ByteRate}.");
        return false;
      }

      if (BlockAlign != (NumChannels * BitsPerSample / 8))
      {
        Console.WriteLine($"Invalid block align: {ByteRate}.");
        return false;
      }

      if (BitsPerSample != 16)
      {  // we support only 16 bits per sample
        Console.WriteLine($"Invalid bits per sample: {BitsPerSample}. Expect 16");
        return false;
      }

      return true;
    }
  }

  // It supports only 16-bit, single channel WAVE format.
  // The sample rate can be any value.
  public class WaveReader
  {
    public WaveReader(String fileName)
    {
      if (!File.Exists(fileName))
      {
        throw new ApplicationException($"{fileName} does not exist!");
      }

      using (var stream = File.Open(fileName, FileMode.Open))
      {
        using (var reader = new BinaryReader(stream))
        {
          _header = ReadHeader(reader);

          if (!_header.Validate())
          {
            throw new ApplicationException($"Invalid wave file ${fileName}");
          }

          SkipMetaData(reader);

          // now read samples
          // _header.SubChunk2Size contains number of bytes in total.
          // we assume each sample is of type int16
          byte[] buffer = reader.ReadBytes(_header.SubChunk2Size);
          short[] samples_int16 = new short[_header.SubChunk2Size / 2];
          Buffer.BlockCopy(buffer, 0, samples_int16, 0, buffer.Length);

          _samples = new float[samples_int16.Length];

          for (var i = 0; i < samples_int16.Length; ++i)
          {
            _samples[i] = samples_int16[i] / 32768.0F;
          }
        }
      }
    }

    private static WaveHeader ReadHeader(BinaryReader reader)
    {
      byte[] bytes = reader.ReadBytes(Marshal.SizeOf(typeof(WaveHeader)));

      GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
      WaveHeader header = (WaveHeader)Marshal.PtrToStructure(handle.AddrOfPinnedObject(), typeof(WaveHeader))!;
      handle.Free();

      return header;
    }

    private void SkipMetaData(BinaryReader reader)
    {
      var bs = reader.BaseStream;

      Int32 subChunk2ID = _header.SubChunk2ID;
      Int32 subChunk2Size = _header.SubChunk2Size;

      while (bs.Position != bs.Length && subChunk2ID != 0x61746164)
      {
        bs.Seek(subChunk2Size, SeekOrigin.Current);
        subChunk2ID = reader.ReadInt32();
        subChunk2Size = reader.ReadInt32();
      }
      _header.SubChunk2ID = subChunk2ID;
      _header.SubChunk2Size = subChunk2Size;
    }

    private WaveHeader _header;

    // Samples are normalized to the range [-1, 1]
    private float[] _samples;

    public int SampleRate => _header.SampleRate;
    public float[] Samples => _samples;

    public static void Test(String fileName)
    {
      WaveReader reader = new WaveReader(fileName);
      Console.WriteLine($"samples length: {reader.Samples.Length}");
      Console.WriteLine($"samples rate: {reader.SampleRate}");
    }
  }

}
