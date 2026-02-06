package main

import (
	"encoding/binary"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"syscall"
	"time"

	oto "github.com/ebitengine/oto/v3"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	flag "github.com/spf13/pflag"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	config := sherpa.OfflineTtsConfig{}
	sid := 0

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

	flag.Parse()

	if len(flag.Args()) != 1 {
		log.Fatalf("Please provide the text to generate audios")
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

	// Pipe: TTS writes → Oto reads
	pr, pw := io.Pipe()
	player := ctx.NewPlayer(pr)
	player.Play()
	defer player.Close()

	// Ctrl+C handling
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	log.Println("Start generating & playing audio")

	done := make(chan struct{})

	go func() {
		defer close(done)
		tts.GenerateWithCallback(text, sid, 1.0, func(samples []float32) bool {
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

			_, err := pw.Write(buf)
			if err != nil {
				log.Println("audio pipe write error:", err)
				return false
			}
			return true
		})
		pw.Close()
	}()

	select {
	case <-stop:
		log.Println("Interrupted, stopping...")
	case <-done:
		log.Println("TTS finished")
	}

	log.Println("Done")
}
