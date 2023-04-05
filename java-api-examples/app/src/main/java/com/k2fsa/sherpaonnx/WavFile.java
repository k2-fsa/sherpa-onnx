// Wav reader 
//Copyright 2022-2023 by zhaoming
// change double to float
//Source based on https://github.com/Semantive/waveform-android/blob/master/library/src/main/java/com/semantive/waveformandroid/waveform/soundfile/WavFile.java.
package com.k2fsa.sherpaonnx;

import java.io.*;

public class WavFile {
    private enum IOState {READING, WRITING, CLOSED}

    ;
    private final static int BUFFER_SIZE = 4096;

    private final static int FMT_CHUNK_ID = 0x20746D66;
    private final static int DATA_CHUNK_ID = 0x61746164;
    private final static int RIFF_CHUNK_ID = 0x46464952;
    private final static int RIFF_TYPE_ID = 0x45564157;

    private File file;                        // File that will be read from or written to
    private IOState ioState;                // Specifies the IO State of the Wav File (used for snaity checking)
    private int bytesPerSample;            // Number of bytes required to store a single sample
    private long numFrames;                    // Number of frames within the data section
    private FileOutputStream oStream;    // Output stream used for writting data
    private FileInputStream iStream;        // Input stream used for reading data
    private float floatScale;                // Scaling factor used for int <-> float conversion
    private float floatOffset;            // Offset factor used for int <-> float conversion
    private boolean wordAlignAdjust;        // Specify if an extra byte at the end of the data chunk is required for word alignment

    // Wav Header
    private int numChannels;                // 2 bytes unsigned, 0x0001 (1) to 0xFFFF (65,535)
    private long sampleRate;                // 4 bytes unsigned, 0x00000001 (1) to 0xFFFFFFFF (4,294,967,295)
    // Although a java int is 4 bytes, it is signed, so need to use a long
    private int blockAlign;                    // 2 bytes unsigned, 0x0001 (1) to 0xFFFF (65,535)
    private int validBits;                    // 2 bytes unsigned, 0x0002 (2) to 0xFFFF (65,535)

    // Buffering
    private byte[] buffer;                    // Local buffer used for IO
    private int bufferPointer;                // Points to the current position in local buffer
    private int bytesRead;                    // Bytes read after last read into local buffer
    private long frameCounter;                // Current number of frames read or written

    // Cannot instantiate WavFile directly, must either use newWavFile() or openWavFile()
    private WavFile() {
        buffer = new byte[BUFFER_SIZE];
    }

    public int getNumChannels() {
        return numChannels;
    }

    public long getNumFrames() {
        return numFrames;
    }

    public long getFramesRemaining() {
        return numFrames - frameCounter;
    }

    public long getSampleRate() {
        return sampleRate;
    }

    public int getValidBits() {
        return validBits;
    }


    public static WavFile openWavFile(File file) throws IOException {
        // Instantiate new Wavfile and store the file reference
        WavFile wavFile = new WavFile();
        wavFile.file = file;

        // Create a new file input stream for reading file data
        wavFile.iStream = new FileInputStream(file);

        // Read the first 12 bytes of the file
        int bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 12);
        if (bytesRead != 12) throw new IOException("Not enough wav file bytes for header");

        // Extract parts from the header
        long riffChunkID = getLE(wavFile.buffer, 0, 4);
        long chunkSize = getLE(wavFile.buffer, 4, 4);
        long riffTypeID = getLE(wavFile.buffer, 8, 4);

        // Check the header bytes contains the correct signature
        if (riffChunkID != RIFF_CHUNK_ID) throw new IOException("Invalid Wav Header data, incorrect riff chunk ID");
        if (riffTypeID != RIFF_TYPE_ID) throw new IOException("Invalid Wav Header data, incorrect riff type ID");

        // Check that the file size matches the number of bytes listed in header
        if (file.length() != chunkSize + 8) {
            throw new IOException("Header chunk size (" + chunkSize + ") does not match file size (" + file.length() + ")");
        }

        boolean foundFormat = false;
        boolean foundData = false;

        // Search for the Format and Data Chunks
        while (true) {
            // Read the first 8 bytes of the chunk (ID and chunk size)
            bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 8);
            if (bytesRead == -1) throw new IOException("Reached end of file without finding format chunk");
            if (bytesRead != 8) throw new IOException("Could not read chunk header");

            // Extract the chunk ID and Size
            long chunkID = getLE(wavFile.buffer, 0, 4);
            chunkSize = getLE(wavFile.buffer, 4, 4);

            // Word align the chunk size
            // chunkSize specifies the number of bytes holding data. However,
            // the data should be word aligned (2 bytes) so we need to calculate
            // the actual number of bytes in the chunk
            long numChunkBytes = (chunkSize % 2 == 1) ? chunkSize + 1 : chunkSize;

            if (chunkID == FMT_CHUNK_ID) {
                // Flag that the format chunk has been found
                foundFormat = true;

                // Read in the header info
                bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 16);

                // Check this is uncompressed data
                int compressionCode = (int) getLE(wavFile.buffer, 0, 2);
                if (compressionCode != 1)
                    throw new IOException("Compression Code " + compressionCode + " not supported");

                // Extract the format information
                wavFile.numChannels = (int) getLE(wavFile.buffer, 2, 2);
                wavFile.sampleRate = getLE(wavFile.buffer, 4, 4);
                wavFile.blockAlign = (int) getLE(wavFile.buffer, 12, 2);
                wavFile.validBits = (int) getLE(wavFile.buffer, 14, 2);

                if (wavFile.numChannels == 0)
                    throw new IOException("Number of channels specified in header is equal to zero");
                if (wavFile.blockAlign == 0) throw new IOException("Block Align specified in header is equal to zero");
                if (wavFile.validBits < 2) throw new IOException("Valid Bits specified in header is less than 2");
                if (wavFile.validBits > 64)
                    throw new IOException("Valid Bits specified in header is greater than 64, this is greater than a long can hold");

                // Calculate the number of bytes required to hold 1 sample
                wavFile.bytesPerSample = (wavFile.validBits + 7) / 8;
                if (wavFile.bytesPerSample * wavFile.numChannels != wavFile.blockAlign)
                    throw new IOException("Block Align does not agree with bytes required for validBits and number of channels");

                // Account for number of format bytes and then skip over
                // any extra format bytes
                numChunkBytes -= 16;
                if (numChunkBytes > 0) wavFile.iStream.skip(numChunkBytes);
            } else if (chunkID == DATA_CHUNK_ID) {
                // Check if we've found the format chunk,
                // If not, throw an exception as we need the format information
                // before we can read the data chunk
                if (foundFormat == false) throw new IOException("Data chunk found before Format chunk");

                // Check that the chunkSize (wav data length) is a multiple of the
                // block align (bytes per frame)
                if (chunkSize % wavFile.blockAlign != 0)
                    throw new IOException("Data Chunk size is not multiple of Block Align");

                // Calculate the number of frames
                wavFile.numFrames = chunkSize / wavFile.blockAlign;

                // Flag that we've found the wave data chunk
                foundData = true;

                break;
            } else {
                // If an unknown chunk ID is found, just skip over the chunk data
                wavFile.iStream.skip(numChunkBytes);
            }
        }

        // Throw an exception if no data chunk has been found
        if (foundData == false) throw new IOException("Did not find a data chunk");

        // Calculate the scaling factor for converting to a normalised float
        if (wavFile.validBits > 8) {
            // If more than 8 validBits, data is signed
            // Conversion required dividing by magnitude of max negative value
            wavFile.floatOffset = 0;
            wavFile.floatScale = 1 << (wavFile.validBits - 1);
        } else {
            // Else if 8 or less validBits, data is unsigned
            // Conversion required dividing by max positive value
            wavFile.floatOffset = -1;
            wavFile.floatScale = 0.5F * ((1 << wavFile.validBits) - 1);
        }

        wavFile.bufferPointer = 0;
        wavFile.bytesRead = 0;
        wavFile.frameCounter = 0;
        wavFile.ioState = IOState.READING;

        return wavFile;
    }

    // Get and Put little endian data from local buffer
    // ------------------------------------------------
    private static long getLE(byte[] buffer, int pos, int numBytes) {
        numBytes--;
        pos += numBytes;

        long val = buffer[pos] & 0xFF;
        for (int b = 0; b < numBytes; b++) val = (val << 8) + (buffer[--pos] & 0xFF);

        return val;
    }

    private static void putLE(long val, byte[] buffer, int pos, int numBytes) {
        for (int b = 0; b < numBytes; b++) {
            buffer[pos] = (byte) (val & 0xFF);
            val >>= 8;
            pos++;
        }
    }

    // Sample Writing and Reading
    // --------------------------
    private void writeSample(long val) throws IOException {
        for (int b = 0; b < bytesPerSample; b++) {
            if (bufferPointer == BUFFER_SIZE) {
                oStream.write(buffer, 0, BUFFER_SIZE);
                bufferPointer = 0;
            }

            buffer[bufferPointer] = (byte) (val & 0xFF);
            val >>= 8;
            bufferPointer++;
        }
    }

    private long readSample() throws IOException, IOException {
        long val = 0;

        for (int b = 0; b < bytesPerSample; b++) {
            if (bufferPointer == bytesRead) {
                int read = iStream.read(buffer, 0, BUFFER_SIZE);
                if (read == -1) throw new IOException("Not enough data available");
                bytesRead = read;
                bufferPointer = 0;
            }

            int v = buffer[bufferPointer];
            if (b < bytesPerSample - 1 || bytesPerSample == 1) v &= 0xFF;
            val += v << (b * 8);

            bufferPointer++;
        }

        return val;
    }

    // Integer
    // -------
    public int readFrames(int[] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(int[] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                sampleBuffer[offset] = (int) readSample();
                offset++;
            }

            frameCounter++;
        }

        return numFramesToRead;
    }

    public int readFrames(int[][] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(int[][] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) sampleBuffer[c][offset] = (int) readSample();

            offset++;
            frameCounter++;
        }

        return numFramesToRead;
    }

    public int writeFrames(int[] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(int[] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                writeSample(sampleBuffer[offset]);
                offset++;
            }

            frameCounter++;
        }

        return numFramesToWrite;
    }

    public int writeFrames(int[][] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(int[][] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) writeSample(sampleBuffer[c][offset]);

            offset++;
            frameCounter++;
        }

        return numFramesToWrite;
    }

    // Long
    // ----
    public int readFrames(long[] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(long[] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                sampleBuffer[offset] = readSample();
                offset++;
            }

            frameCounter++;
        }

        return numFramesToRead;
    }

    public int readFrames(long[][] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(long[][] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) sampleBuffer[c][offset] = readSample();

            offset++;
            frameCounter++;
        }

        return numFramesToRead;
    }

    public int writeFrames(long[] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(long[] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                writeSample(sampleBuffer[offset]);
                offset++;
            }

            frameCounter++;
        }

        return numFramesToWrite;
    }

    public int writeFrames(long[][] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(long[][] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) writeSample(sampleBuffer[c][offset]);

            offset++;
            frameCounter++;
        }

        return numFramesToWrite;
    }

    // Double
    // ------
    public int readFrames(float[] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(float[] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                sampleBuffer[offset] = floatOffset + (float) readSample() / floatScale;
                offset++;
            }

            frameCounter++;
        }

        return numFramesToRead;
    }

    public int readFrames(float[][] sampleBuffer, int numFramesToRead) throws IOException, IOException {
        return readFrames(sampleBuffer, 0, numFramesToRead);
    }

    public int readFrames(float[][] sampleBuffer, int offset, int numFramesToRead) throws IOException, IOException {
        if (ioState != IOState.READING) throw new IOException("Cannot read from WavFile instance");

        for (int f = 0; f < numFramesToRead; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++)
                sampleBuffer[c][offset] = floatOffset + (float) readSample() / floatScale;

            offset++;
            frameCounter++;
        }

        return numFramesToRead;
    }

    public int writeFrames(float[] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(float[] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++) {
                writeSample((long) (floatScale * (floatOffset + sampleBuffer[offset])));
                offset++;
            }

            frameCounter++;
        }

        return numFramesToWrite;
    }

    public int writeFrames(float[][] sampleBuffer, int numFramesToWrite) throws IOException, IOException {
        return writeFrames(sampleBuffer, 0, numFramesToWrite);
    }

    public int writeFrames(float[][] sampleBuffer, int offset, int numFramesToWrite) throws IOException, IOException {
        if (ioState != IOState.WRITING) throw new IOException("Cannot write to WavFile instance");

        for (int f = 0; f < numFramesToWrite; f++) {
            if (frameCounter == numFrames) return f;

            for (int c = 0; c < numChannels; c++)
                writeSample((long) (floatScale * (floatOffset + sampleBuffer[c][offset])));

            offset++;
            frameCounter++;
        }

        return numFramesToWrite;
    }


    public void close() throws IOException {
        // Close the input stream and set to null
        if (iStream != null) {
            iStream.close();
            iStream = null;
        }

        if (oStream != null) {
            // Write out anything still in the local buffer
            if (bufferPointer > 0) oStream.write(buffer, 0, bufferPointer);

            // If an extra byte is required for word alignment, add it to the end
            if (wordAlignAdjust) oStream.write(0);

            // Close the stream and set to null
            oStream.close();
            oStream = null;
        }

        // Flag that the stream is closed
        ioState = IOState.CLOSED;
    }

    public void display() {
        display(System.out);
    }

    public void display(PrintStream out) {
        out.printf("File: %s\n", file);
        out.printf("Channels: %d, Frames: %d\n", numChannels, numFrames);
        out.printf("IO State: %s\n", ioState);
        out.printf("Sample Rate: %d, Block Align: %d\n", sampleRate, blockAlign);
        out.printf("Valid Bits: %d, Bytes per sample: %d\n", validBits, bytesPerSample);
    }


}
