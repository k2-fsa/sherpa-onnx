package main

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	oto "github.com/ebitengine/oto/v3"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

type pcmBuffer struct {
	mu       sync.Mutex
	queue    [][]byte
	finished bool
	started  chan struct{} // closed on first callback
	once     sync.Once
}

func newPCMBuffer() *pcmBuffer {
	return &pcmBuffer{
		started: make(chan struct{}),
	}
}

func (b *pcmBuffer) Push(p []byte) {
	b.once.Do(func() {
		close(b.started)
	})

	b.mu.Lock()
	b.queue = append(b.queue, p)
	b.mu.Unlock()
}

func (b *pcmBuffer) Finish() {
	b.once.Do(func() {
		close(b.started)
	})

	b.mu.Lock()
	b.finished = true
	b.mu.Unlock()
}

type pcmReader struct {
	buf  *pcmBuffer
	done chan struct{}
	once sync.Once
}

func (r *pcmReader) Read(p []byte) (int, error) {
	<-r.buf.started

	r.buf.mu.Lock()
	defer r.buf.mu.Unlock()

	// 2) Have audio
	if len(r.buf.queue) > 0 {
		chunk := r.buf.queue[0]
		n := copy(p, chunk)

		if n == len(chunk) {
			r.buf.queue = r.buf.queue[1:]
		} else {
			r.buf.queue[0] = chunk[n:]
		}
		return n, nil
	}

	// 3) Finished → EOF
	if r.buf.finished {
		r.once.Do(func() { close(r.done) })
		return 0, io.EOF
	}

	// 4) Gap → silence
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var referenceAudio string = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
	var outputFilename string = "./generated.wav"
	var voiceEmbeddingCacheCapacity int = 50
	var seed int = -1

	text := `Today as always, men fall into two groups: slaves and free men.
Whoever does not have two-thirds of his day for himself, is a slave,
whatever he may be: a statesman, a businessman, an official, or a scholar.`

	flag.StringVar(&referenceAudio, "reference-audio", referenceAudio, "Path to the reference audio")
	flag.StringVar(&text, "text", text, "Text to be synthesized")
	flag.StringVar(&outputFilename, "output-filename", outputFilename, "File to save the generated audio")
	flag.IntVar(&voiceEmbeddingCacheCapacity, "voice-embedding-cache-capacity", voiceEmbeddingCacheCapacity, "Voice embedding cache capacity (default: 50)")
	flag.IntVar(&seed, "seed", seed, "Random seed for reproducibility (default: -1, random)")
	flag.Parse()

	// ---------------- config ----------------
	var config sherpa.OfflineTtsConfig

	config.Model.Pocket.LmFlow =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx"
	config.Model.Pocket.LmMain =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx"
	config.Model.Pocket.Encoder =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx"
	config.Model.Pocket.Decoder =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx"
	config.Model.Pocket.TextConditioner =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx"
	config.Model.Pocket.VocabJson =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json"
	config.Model.Pocket.TokenScoresJson =
		"./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json"
	config.Model.Pocket.VoiceEmbeddingCacheCapacity = voiceEmbeddingCacheCapacity

	config.Model.NumThreads = 2
	config.Model.Debug = 0
	config.Model.Provider = "cpu"

	log.Println("Creating Offline TTS")
	tts := sherpa.NewOfflineTts(&config)
	if tts == nil {
		log.Fatal("Failed to create OfflineTts")
	}
	defer sherpa.DeleteOfflineTts(tts)

	wave := sherpa.ReadWave(referenceAudio)
	if wave == nil {
		log.Fatal("Failed to read reference wav:", referenceAudio)
	}

	var cfg sherpa.GenerationConfig
	cfg.ReferenceAudio = wave.Samples
	cfg.ReferenceSampleRate = wave.SampleRate

	// Build extra config with optional seed
	extraMap := map[string]interface{}{
		"max_reference_audio_len": 10,
		"temperature":             0.7,
	}
	if seed >= 0 {
		extraMap["seed"] = seed
	}
	extraBytes, _ := json.Marshal(extraMap)
	cfg.Extra = json.RawMessage(extraBytes)

	log.Println("Start generating")

	ctx, ready, err := oto.NewContext(&oto.NewContextOptions{
		SampleRate:   tts.SampleRate(),
		ChannelCount: 1,
		Format:       oto.FormatSignedInt16LE,
	})
	if err != nil {
		log.Fatal(err)
	}
	<-ready

	pcmBuf := newPCMBuffer()

	reader := &pcmReader{
		buf:  pcmBuf,
		done: make(chan struct{}),
	}

	player := ctx.NewPlayer(reader)
	player.Play()
	defer player.Close()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	var generated *sherpa.GeneratedAudio

	start := time.Now()

	go func() {
		defer pcmBuf.Finish()

		generated = tts.GenerateWithConfig(
			text,
			&cfg,
			func(samples []float32, progress float32) bool {
				log.Printf("Progress: %.1f%%", progress*100)

				buf := make([]byte, len(samples)*2)
				for i, s := range samples {
					if s > 1 {
						s = 1
					} else if s < -1 {
						s = -1
					}
					v := int16(math.Round(float64(s * 32767)))
					binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
				}

				pcmBuf.Push(buf)
				return true
			},
		)

		log.Println("TTS generation finished in", time.Since(start))
	}()

	select {
	case <-stop:
		log.Println("Interrupted")
	case <-reader.done:
		log.Println("Playback finished")
	}

	if generated != nil {
		if ok := generated.Save(outputFilename); !ok {
			log.Println("Failed to save audio")
		} else {
			log.Println("Saved generated audio to", outputFilename)
		}
	}

	// let remaining audio drain
	time.Sleep(800 * time.Millisecond)

	log.Println("Done")
}
