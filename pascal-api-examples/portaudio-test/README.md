# Introduction

[test-play.pas](./test-play.pas) and [test-record.pas](./test-record.pas)
require that the portaudio library is installed on your system.


On macOS, you can use

```bash
brew install portaudio
```

On macOS with an Intel CPU, it installs `portaudio` into
`/usr/local/Cellar/portaudio/19.7.0`.

On macOS with Apple silicon (arm64), e.g., M1/M2/M3/M4, it installs
`portaudio` into `/opt/homebrew/Cellar/portaudio/19.7.0`.

The run scripts in this directory and in [../tts](../tts) pass both
locations to the Free Pascal compiler via `-Fl`, so they work on Intel and
Apple silicon Macs. If your Homebrew installs a different portaudio
version, please adjust the paths in the run scripts accordingly.
