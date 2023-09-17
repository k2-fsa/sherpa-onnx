# File description

- [./http_server.py](./http_server.py) It defines which files to server.
  Files are saved in [./web](./web).
- [non_streaming_server.py](./non_streaming_server.py) WebSocket server for
  non-streaming models.
- [vad-remove-non-speech-segments.py](./vad-remove-non-speech-segments.py) It uses
  [silero-vad](https://github.com/snakers4/silero-vad) to remove non-speech
  segments and concatenate all speech segments into a single one.
