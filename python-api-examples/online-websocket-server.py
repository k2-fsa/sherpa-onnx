#!/usr/bin/env python3
# Copyright      2025  Uniphore

'''
Real-time speech recognition server using WebSockets.
Python API interface to start the server.
Python wrapper around implementation of online-websocket-server.cc in C++.

(1) Download streaming transducer model

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
rm sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

(2) Starting websocket server using the downloaded model

python3 ./python-api-examples/online-websocket-server.py \
  --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
  --max-batch-size=5 \
  --loop-interval-ms=10
  
'''
import argparse
import sys
import signal
from sherpa_onnx import OnlineWebSocketServer

def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

# Bind SIGINT to signal_handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":    
    args = sys.argv[:]
    OnlineWebSocketServer(server_args=args)