// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

/**
 * Forced aligner: given a wave file and its transcript, it computes the
 * start/end time of every phone and word.
 *
 * <pre>
 * ForcedAlignerConfig config = new ForcedAlignerConfig();
 * config.modelDir = "/path/to/english-mfa-v3.1.0";
 * config.lexicon = "/path/to/english-mfa-v3.1.0/lexicon.txt";
 *
 * ForcedAligner aligner = new ForcedAligner(config);
 * OfflineStream stream = aligner.createStream();
 * stream.acceptWaveform(samples, sampleRate); // float32, normalized to [-1, 1]
 * ForcedAlignmentResult result = aligner.compute(stream, text);
 * System.out.println(result.toTextGrid());
 * </pre>
 */
public class ForcedAligner {
    private long ptr;

    /**
     * Creates a forced aligner from a model directory and a pronunciation
     * dictionary. It loads the acoustic model, the tree and the lexicon.
     *
     * @param config configuration for forced alignment
     * @throws IllegalArgumentException if the native aligner cannot be
     *     created (e.g. a missing model file; see stderr for the reason)
     */
    public ForcedAligner(ForcedAlignerConfig config) {
        LibraryLoader.maybeLoad();
        ptr = newFromFile(config);
        if (ptr == 0) {
            throw new IllegalArgumentException(
                    "Invalid ForcedAlignerConfig: failed to create native ForcedAligner");
        }
    }

    /**
     * Creates an utterance stream. Accept a waveform with
     * {@link OfflineStream#acceptWaveform}, then pass the stream to
     * {@link #compute}.
     *
     * @return a new {@link OfflineStream}
     */
    public OfflineStream createStream() {
        long p = createStream(ptr);
        return new OfflineStream(p);
    }

    /**
     * Computes the alignment of one utterance.
     *
     * @param stream a stream created by {@link #createStream} and filled
     *     with a waveform
     * @param text the transcript of the whole utterance. Hanzi text is
     *     tokenized per character (multi-character dictionary keys give
     *     pronunciation context); anything else is split on whitespace.
     *     Punctuation is ignored
     * @return the alignment result; empty (no tiers) with a warning on
     *     stderr when the stream is null, the text is empty, the stream has
     *     no features, or word boundaries cannot be recovered
     */
    public ForcedAlignmentResult compute(OfflineStream stream, String text) {
        return compute(ptr, stream.getPtr(), text);
    }

    @Override
    protected void finalize() throws Throwable {
        release();
    }

    // You'd better call it manually if it is not used anymore
    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    private native void delete(long ptr);

    private native long newFromFile(ForcedAlignerConfig config);

    private native long createStream(long ptr);

    private native ForcedAlignmentResult compute(long ptr, long streamPtr,
                                                 String text);
}
