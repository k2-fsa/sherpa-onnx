package main

import (
	"encoding/binary"
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

	config := sherpa.OfflineTtsConfig{}
	sid := 0
	filename := "./generated.wav"

	flag.StringVar(&config.Model.Vits.Model, "vits-model", "", "Path to the vits ONNX model")
	flag.StringVar(&config.Model.Vits.Lexicon, "vits-lexicon", "", "Path to lexicon.txt")
	flag.StringVar(&config.Model.Vits.Tokens, "vits-tokens", "", "Path to tokens.txt")
	flag.StringVar(&config.Model.Vits.DataDir, "vits-data-dir", "", "Path to espeak-ng-data")

	flag.Float32Var(&config.Model.Vits.NoiseScale, "vits-noise-scale", 0.667, "noise_scale for VITS")
	flag.Float32Var(&config.Model.Vits.NoiseScaleW, "vits-noise-scale-w", 0.8, "noise_scale_w for VITS")
	flag.Float32Var(&config.Model.Vits.LengthScale, "vits-length-scale", 1.0, "length_scale for VITS. small -> faster in speech speed; large -> slower")

	flag.StringVar(&config.Model.Matcha.AcousticModel, "matcha-acoustic-model", "", "Path to the matcha acoustic model")
	flag.StringVar(&config.Model.Matcha.Vocoder, "matcha-vocoder", "", "Path to the matcha vocoder model")
	flag.StringVar(&config.Model.Matcha.Lexicon, "matcha-lexicon", "", "Path to lexicon.txt")
	flag.StringVar(&config.Model.Matcha.Tokens, "matcha-tokens", "", "Path to tokens.txt")
	flag.StringVar(&config.Model.Matcha.DataDir, "matcha-data-dir", "", "Path to espeak-ng-data")

	flag.Float32Var(&config.Model.Matcha.NoiseScale, "matcha-noise-scale", 0.667, "noise_scale for Matcha")
	flag.Float32Var(&config.Model.Matcha.LengthScale, "matcha-length-scale", 1.0, "length_scale for Matcha. small -> faster in speech speed; large -> slower")

	flag.StringVar(&config.Model.Kokoro.Model, "kokoro-model", "", "Path to the Kokoro ONNX model")
	flag.StringVar(&config.Model.Kokoro.Voices, "kokoro-voices", "", "Path to voices.bin for Kokoro")
	flag.StringVar(&config.Model.Kokoro.Tokens, "kokoro-tokens", "", "Path to tokens.txt for Kokoro")
	flag.StringVar(&config.Model.Kokoro.DataDir, "kokoro-data-dir", "", "Path to espeak-ng-data for Kokoro")
	flag.StringVar(&config.Model.Kokoro.Lexicon, "kokoro-lexicon", "", "Path to lexicon files for Kokoro")
	flag.Float32Var(&config.Model.Kokoro.LengthScale, "kokoro-length-scale", 1.0, "length_scale for Kokoro. small -> faster in speech speed; large -> slower")

	flag.StringVar(&config.Model.Kitten.Model, "kitten-model", "", "Path to the kitten ONNX model")
	flag.StringVar(&config.Model.Kitten.Voices, "kitten-voices", "", "Path to voices.bin for kitten")
	flag.StringVar(&config.Model.Kitten.Tokens, "kitten-tokens", "", "Path to tokens.txt for kitten")
	flag.StringVar(&config.Model.Kitten.DataDir, "kitten-data-dir", "", "Path to espeak-ng-data for kitten")
	flag.Float32Var(&config.Model.Kitten.LengthScale, "kitten-length-scale", 1.0, "length_scale for kitten. small -> faster in speech speed; large -> slower")

	flag.IntVar(&config.Model.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.IntVar(&config.Model.Debug, "debug", 0, "Whether to show debug message")
	flag.StringVar(&config.Model.Provider, "provider", "cpu", "Provider to use")
	flag.StringVar(&config.RuleFsts, "tts-rule-fsts", "", "Path to rule.fst")
	flag.StringVar(&config.RuleFars, "tts-rule-fars", "", "Path to rule.far")
	flag.IntVar(&config.MaxNumSentences, "tts-max-num-sentences", 1, "Batch size")

	flag.IntVar(&sid, "sid", 0, "Speaker ID. Used only for multi-speaker models")
	flag.StringVar(&filename, "output-filename", "./generated.wav", "Output wav filename")

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide the text to generate audio")
	}

	text := flag.Arg(0)

	log.Println("Input text:", text)
	log.Println("Speaker ID:", sid)

	log.Println("Initializing model (may take several seconds)")
	tts := sherpa.NewOfflineTts(&config)
	defer sherpa.DeleteOfflineTts(tts)
	log.Println("Model created!")

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

		generated = tts.GenerateWithProgressCallback(
			text,
			sid,
			1.0,
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
		if ok := generated.Save(filename); !ok {
			log.Println("Failed to save audio")
		} else {
			log.Println("Saved generated audio to", filename)
		}
	}

	// let remaining audio drain
	time.Sleep(800 * time.Millisecond)

	log.Println("Done")
}
