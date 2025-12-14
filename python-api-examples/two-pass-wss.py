#!/usr/bin/env python3
# Copyright (c) 2025 Minghu Wang
"""

A two-pass streaming ASR server with WebSocket support. This server implements
a two-pass recognition strategy where the first pass uses a fast streaming model
for real-time recognition, and the second pass uses a more accurate offline model
to refine the results.

The first pass provides immediate feedback to users, while the second pass
improves accuracy by re-processing the complete utterance with a more powerful
model.

It supports multiple clients sending audio simultaneously and provides
real-time transcription results.

Usage:
    ./two-pass-wss.py --help

Example:

(1) Without a certificate

python3 ./python-api-examples/two-pass-wss.py \
  --paraformer-encoder ./sherpa-onnx-paraformer-zh-2023-09-18/encoder.onnx \
  --paraformer-decoder ./sherpa-onnx-paraformer-zh-2023-09-18/decoder.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-18/tokens.txt \
  --second-sense-voice ./sherpa-onnx-sense-voice-zh-2023-09-18/model.onnx \
  --second-tokens ./sherpa-onnx-sense-voice-zh-2023-09-18/tokens.txt

(2) With a certificate

(a) Generate a certificate first:

    cd python-api-examples/web
    ./generate-certificate.py
    cd ../..

(b) Start the server

python3 ./python-api-examples/two-pass-wss.py \
  --paraformer-encoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx \
  --paraformer-decoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.onnx \
  --tokens ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --second-sense-voice ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
  --second-tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --certificate ./python-api-examples/web/cert.pem

Please refer to
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
to download pre-trained models.
"""

import argparse
import asyncio
import http
import json
import logging
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sherpa_onnx
import websockets

def setup_logger(
    log_filename: str,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    log_filename = f"{log_filename}-{date_time}.txt"

    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default="",
        help="Path to the transducer decoder model.",
    )


    parser.add_argument(
        "--second-tokens",
        type=str,
        default="",
        help="Path to the second pass tokens.txt",
    )

    parser.add_argument(
        "--second-sense-voice",
        type=str,
        default="",
        help="Path to the second pass sense voice model.",
    )

    parser.add_argument(
        "--paraformer-encoder",
        type=str,
        default="",
        help="Path to the paraformer encoder model",
    )

    parser.add_argument(
        "--paraformer-decoder",
        type=str,
        default="",
        help="Path to the paraformer decoder model.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the data used to train the model. "
        "Caution: If your input sound files have a different sampling rate, "
        "we will do resampling inside",
    )

    parser.add_argument(
        "--feat-dim",
        type=int,
        default=80,
        help="Feature dimension of the model",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Decoding method to use. Current supported methods are:
        - greedy_search
        - modified_beam_search
        """,
    )

    add_modified_beam_search_args(parser)


def add_hotwords_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )
    parser.add_argument(
        "--modeling-unit",
        type=str,
        default='cjkchar',
        help="""
        The modeling unit of the used model. Current supported units are:
        - cjkchar(for Chinese)
        - bpe(for English like languages)
        - cjkchar+bpe(for multilingual models)
        """,
    )
    parser.add_argument(
        "--bpe-vocab",
        type=str,
        default='',
        help="""
        The bpe vocabulary generated by sentencepiece toolkit. 
        It is only used when modeling-unit is bpe or cjkchar+bpe.
        if you can't find bpe.vocab in the model directory, please run:
        python script/export_bpe_vocab.py --bpe-model exp/bpe.model
        """,
    )


def add_modified_beam_search_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )

def add_blank_penalty_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

def add_endpointing_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--rule1-min-trailing-silence",
        type=float,
        default=2.4,
        help="""This endpointing rule1 requires duration of trailing silence
        in seconds) to be >= this value""",
    )

    parser.add_argument(
        "--rule2-min-trailing-silence",
        type=float,
        default=1.2,
        help="""This endpointing rule2 requires duration of trailing silence in
        seconds) to be >= this value.""",
    )

    parser.add_argument(
        "--rule3-min-utterance-length",
        type=float,
        default=20,
        help="""This endpointing rule3 requires utterance-length (in seconds)
        to be >= this value.""",
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_model_args(parser)
    add_decoding_args(parser)
    add_endpointing_args(parser)
    add_hotwords_args(parser)
    add_blank_penalty_args(parser)

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--nn-pool-size",
        type=int,
        default=1,
        help="Number of threads for NN computation and decoding.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=3,
        help="""Max batch size for computation. Note if there are not enough
        requests in the queue, it will wait for max_wait_ms time. After that,
        even if there are not enough requests, it still sends the
        available requests in the queue for computation.
        """,
    )

    parser.add_argument(
        "--max-wait-ms",
        type=float,
        default=10,
        help="""Max time in millisecond to wait to build batches for inference.
        If there are not enough requests in the stream queue to build a batch
        of max_batch_size, it waits up to this time before fetching available
        requests for computation.
        """,
    )

    parser.add_argument(
        "--max-message-size",
        type=int,
        default=(1 << 20),
        help="""Max message size in bytes.
        The max size per message cannot exceed this limit.
        """,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=32,
        help="Max number of messages in the queue for each connection.",
    )

    parser.add_argument(
        "--max-active-connections",
        type=int,
        default=200,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads to run the neural network model",
    )

    parser.add_argument(
        "--second-pass-threads",
        type=int,
        default=2,
        help="Number of threads for second pass processing",
    )

    parser.add_argument(
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use ./web/generate-certificate.py
        to generate the certificate `cert.pem`.
        Note ./web/generate-certificate.py will generate three files but you
        only need to pass the generated cert.pem to this option.
        """,
    )

    return parser.parse_args()

def run_second_pass(
    recognizer: sherpa_onnx.OfflineRecognizer,
    samples: np.ndarray,
    sample_rate: int,
):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    recognizer.decode_stream(stream)

    return stream.result.text

def create_first_pass_recognizer(args) -> sherpa_onnx.OnlineRecognizer:
    recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=args.tokens,
            encoder=args.paraformer_encoder,
            decoder=args.paraformer_decoder,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feat_dim,
            decoding_method=args.decoding_method,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=args.rule1_min_trailing_silence,
            rule2_min_trailing_silence=args.rule2_min_trailing_silence,
            rule3_min_utterance_length=args.rule3_min_utterance_length,
            provider=args.provider,
        )
    return recognizer


def create_second_pass_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.second_sense_voice,
            tokens=args.second_tokens,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            use_itn=True,
            decoding_method="greedy_search",
        )
    return recognizer


def format_timestamps(timestamps: List[float]) -> List[str]:
    return ["{:.3f}".format(t) for t in timestamps]


class StreamingServer(object):
    def __init__(
        self,
        first_pass_recognizer: sherpa_onnx.OnlineRecognizer,
        second_pass_recognizer: sherpa_onnx.OfflineRecognizer,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        second_pass_threads: int = 2,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          first_pass_recognizer:
            An instance of online recognizer for first pass.
          second_pass_recognizer:
            An instance of offline recognizer for second pass.
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          max_batch_size:
            Max batch size for inference.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        self.first_pass_recognizer = first_pass_recognizer
        self.second_pass_recognizer = second_pass_recognizer

        self.certificate = certificate

        self.nn_pool_size = nn_pool_size
        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.second_pass_pool = ThreadPoolExecutor(
            max_workers=second_pass_threads,
            thread_name_prefix="second_pass",
        )

        self.stream_queue = asyncio.Queue()

        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

        self.sample_rate = int(self.first_pass_recognizer.config.feat_config.sampling_rate)

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the neural network model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()

                    assert self.first_pass_recognizer.is_ready(item[0])

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.first_pass_recognizer.decode_streams,
                stream_list,
            )

            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(
        self,
        stream: sherpa_onnx.OnlineStream,
    ) -> None:
        """Put the stream into the queue and wait it to be processed by the
        consumer task.

        Args:
          stream:
            The stream to be processed. Note: It is changed in-place.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.stream_queue.put((stream, future))
        await future

    async def run_second_pass_async(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Run second-pass recognition asynchronously to avoid blocking.

        Args:
          samples: Audio samples.
          sample_rate: Sampling rate.

        Returns:
          Text result from the second-pass recognition.
        """
        import time
        start_time = time.time()
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.second_pass_pool,
            run_second_pass,
            self.second_pass_recognizer,
            samples,
            sample_rate,
        )
        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Second pass processing completed in {duration:.3f}s for {len(samples)/sample_rate:.2f}s audio")
        
        return result.lower().strip()

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response

    async def run(self, port: int):
        tasks = []
        for i in range(self.nn_pool_size):
            tasks.append(asyncio.create_task(self.stream_consumer_task()))

        if self.certificate:
            logging.info(f"Using certificate: {self.certificate}")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logging.info("No certificate provided")

        try:
            async with websockets.serve(
                self.handle_connection,
                host="",
                port=port,
                max_size=self.max_message_size,
                max_queue=self.max_queue_size,
                process_request=self.process_request,
                ssl=ssl_context,
            ):
                logging.info(f"Started server on port {port}")
                await asyncio.Future()  # run forever
        finally:
            logging.info("Shutting down thread pools...")
            self.nn_pool.shutdown(wait=True)
            self.second_pass_pool.shutdown(wait=True)
            logging.info("Thread pools shut down successfully")

        await asyncio.gather(*tasks)  # not reachable

    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket)
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"{socket.remote_address} disconnected")
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            logging.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )

    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        stream = self.first_pass_recognizer.create_stream()
        segment = 0
        sample_buffers = []
        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break
            
            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples equal to --sample-rate
            stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
            sample_buffers.append(samples)
            while self.first_pass_recognizer.is_ready(stream):
                await self.compute_and_decode(stream)
                result = self.first_pass_recognizer.get_result(stream)

                message = {
                    "text": result,
                    "segment": segment,
                }
                if self.first_pass_recognizer.is_endpoint(stream):
                    if result:
                        samples_for_2nd_pass = np.concatenate(sample_buffers)
                        sample_buffers = [samples_for_2nd_pass[-8000:]]
                        samples_for_2nd_pass = samples_for_2nd_pass[:-8000]
                        second_pass_result = (
                            await self.run_second_pass_async(
                                samples=samples_for_2nd_pass,
                                sample_rate=self.sample_rate,
                            )
                        )

                        if second_pass_result:
                            message["text"] = second_pass_result
                            message["segment"] = segment
                    else:
                        sample_buffers=[]

                    self.first_pass_recognizer.reset(stream)
                    segment += 1
                await socket.send(json.dumps(message))

        tail_padding = np.zeros(int(self.sample_rate * 0.3)).astype(np.float32)
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=tail_padding)
        stream.input_finished()
        while self.first_pass_recognizer.is_ready(stream):
            await self.compute_and_decode(stream)

        result = self.first_pass_recognizer.get_result(stream)

        message = {
            "text": result,
            "segment": segment,
        }
        await socket.send(json.dumps(message))

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[np.ndarray]:
        """Receive audio samples from WebSocket connection
        
        Args:
          socket: WebSocket connection
        
        Returns:
          Numpy array containing audio samples, or None indicating end of audio
        """
        message = await socket.recv()
        if message == "Done":
            return None
        return np.frombuffer(message, dtype=np.float32)


def check_args(args):
    if args.encoder:
        assert Path(args.encoder).is_file(), f"{args.encoder} does not exist"
        assert Path(args.decoder).is_file(), f"{args.decoder} does not exist"
        assert args.paraformer_encoder is None, args.paraformer_encoder
        assert args.paraformer_decoder is None, args.paraformer_decoder
       
    elif args.paraformer_encoder:
        assert Path(
            args.paraformer_encoder
        ).is_file(), f"{args.paraformer_encoder} does not exist"

        assert Path(
            args.paraformer_decoder
        ).is_file(), f"{args.paraformer_decoder} does not exist"
    else:
        raise ValueError("Please provide a model")

    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.decoding_method not in (
        "greedy_search",
        "modified_beam_search",
    ):
        raise ValueError(f"Unsupported decoding method {args.decoding_method}")

    if args.decoding_method == "modified_beam_search":
        assert args.num_active_paths > 0, args.num_active_paths


def main():
    args = get_args()
    logging.info(vars(args))
    check_args(args)

    first_pass_recognizer = create_first_pass_recognizer(args)
    second_pass_recognizer = create_second_pass_recognizer(args)

    port = args.port
    nn_pool_size = args.nn_pool_size
    max_batch_size = args.max_batch_size
    max_wait_ms = args.max_wait_ms
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    second_pass_threads = args.second_pass_threads
    certificate = args.certificate
    # doc_root = args.doc_root

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    server = StreamingServer(
        first_pass_recognizer=first_pass_recognizer,
        second_pass_recognizer=second_pass_recognizer,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        second_pass_threads=second_pass_threads,
        certificate=certificate,
        # doc_root=doc_root,
    )
    asyncio.run(server.run(port))


if __name__ == "__main__":
    log_filename = "log/log-streaming-server"
    setup_logger(log_filename)
    main()
