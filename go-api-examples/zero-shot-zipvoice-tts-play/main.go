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
	started  chan struct{}
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

	if r.buf.finished {
		r.once.Do(func() { close(r.done) })
		return 0, io.EOF
	}

	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var referenceAudio string = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav"
	var referenceText string = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系."
	var outputFilename string = "./generated.wav"
	var numSteps int = 4
	var minCharInSentence int = 30

	text := "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

	flag.StringVar(&referenceAudio, "reference-audio", referenceAudio, "Path to the reference audio")
	flag.StringVar(&referenceText, "reference-text", referenceText, "Reference text for the reference audio")
	flag.StringVar(&text, "text", text, "Text to be synthesized")
	flag.StringVar(&outputFilename, "output-filename", outputFilename, "File to save the generated audio")
	flag.IntVar(&numSteps, "num-steps", numSteps, "Number of ZipVoice flow-matching steps")
	flag.IntVar(&minCharInSentence, "min-char-in-sentence", minCharInSentence, "Minimum characters in a sentence chunk")
	flag.Parse()

	var config sherpa.OfflineTtsConfig
	config.Model.Zipvoice.Encoder =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx"
	config.Model.Zipvoice.Decoder =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx"
	config.Model.Zipvoice.DataDir =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data"
	config.Model.Zipvoice.Lexicon =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt"
	config.Model.Zipvoice.Tokens =
		"./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt"
	config.Model.Zipvoice.Vocoder = "./vocos_24khz.onnx"

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
	cfg.ReferenceText = referenceText
	cfg.NumSteps = numSteps

	extraMap := map[string]interface{}{
		"min_char_in_sentence": minCharInSentence,
	}
	extraBytes, err := json.Marshal(extraMap)
	if err != nil {
		log.Fatalf("Failed to marshal generation config extra: %v", err)
	}
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

	time.Sleep(800 * time.Millisecond)

	log.Println("Done")
}
