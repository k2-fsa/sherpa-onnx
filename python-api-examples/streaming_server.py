#!/usr/bin/env python3
# Copyright      2022-2023  Xiaomi Corp.
#
"""
A server for streaming ASR recognition. By streaming it means the audio samples
are coming in real-time. You don't need to wait until all audio samples are
captured before sending them for recognition.

It supports multiple clients sending at the same time.

Usage:
    ./streaming_server.py --help

Example:

python3 ./python-api-examples/streaming_server.py \
  --encoder-model ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder-model ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner-model ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt
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
from http_server import HttpServer


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
        "--encoder-model",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder-model",
        type=str,
        required=True,
        help="Path to the decoder model.",
    )

    parser.add_argument(
        "--joiner-model",
        type=str,
        required=True,
        help="Path to the joiner model.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
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


def add_modified_beam_search_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )


def add_endpointing_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use-endpoint",
        type=int,
        default=1,
        help="1 to enable endpoiting. 0 to disable it",
    )

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
        default=50,
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
        default=500,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Sets the number of threads used for interop parallelism (e.g. in JIT interpreter) on CPU.",
    )

    parser.add_argument(
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use sherpa/bin/web/generate-certificate.py
        to generate the certificate `cert.pem`.
        """,
    )

    parser.add_argument(
        "--doc-root",
        type=str,
        default="./python-api-examples/web",
        help="""Path to the web root""",
    )

    return parser.parse_args()


def create_recognizer(args) -> sherpa_onnx.OnlineRecognizer:
    recognizer = sherpa_onnx.OnlineRecognizer(
        tokens=args.tokens,
        encoder=args.encoder_model,
        decoder=args.decoder_model,
        joiner=args.joiner_model,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.num_active_paths,
        enable_endpoint_detection=args.use_endpoint != 0,
        rule1_min_trailing_silence=args.rule1_min_trailing_silence,
        rule2_min_trailing_silence=args.rule2_min_trailing_silence,
        rule3_min_utterance_length=args.rule3_min_utterance_length,
    )

    return recognizer


def format_timestamps(timestamps: List[float]) -> List[str]:
    return ["{:.3f}".format(t) for t in timestamps]


class StreamingServer(object):
    def __init__(
        self,
        recognizer: sherpa_onnx.OnlineRecognizer,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        doc_root: str,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          recognizer:
            An instance of online recognizer.
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
          beam_search_params:
            Dictionary containing all the parameters for beam search.
          online_endpoint_config:
            Config for endpointing.
          doc_root:
            Path to the directory where files like index.html for the HTTP
            server locate.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./sherpa/bin/web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        self.recognizer = recognizer

        self.certificate = certificate
        self.http_server = HttpServer(doc_root)

        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )

        self.stream_queue = asyncio.Queue()
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

        self.sample_rate = int(recognizer.config.feat_config.sampling_rate)
        self.decoding_method = recognizer.config.decoding_method

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the RNN-T model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()

                    assert self.recognizer.is_ready(item[0])

                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.recognizer.decode_streams,
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

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if "sec-websocket-key" not in request_headers:
            # This is a normal HTTP request
            if path == "/":
                path = "/index.html"
            found, response, mime_type = self.http_server.process_request(path)
            if isinstance(response, str):
                response = response.encode("utf-8")

            if not found:
                status = http.HTTPStatus.NOT_FOUND
            else:
                status = http.HTTPStatus.OK
            header = {"Content-Type": mime_type}
            return status, header, response

        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response

    async def run(self, port: int):
        task = asyncio.create_task(self.stream_consumer_task())

        if self.certificate:
            logging.info(f"Using certificate: {self.certificate}")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logging.info("No certificate provided")

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
            ssl=ssl_context,
        ):
            ip_list = ["0.0.0.0", "localhost", "127.0.0.1"]
            ip_list.append(socket.gethostbyname(socket.gethostname()))
            proto = "http://" if ssl_context is None else "https://"
            s = "Please visit one of the following addresses:\n\n"
            for p in ip_list:
                s += "  " + proto + p + f":{port}" "\n"
            logging.info(s)

            await asyncio.Future()  # run forever

        await task  # not reachable

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
        except websockets.exceptions.ConnectionClosedError:
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
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logging.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
        )

        stream = self.recognizer.create_stream()
        segment = 0

        while True:
            samples = await self.recv_audio_samples(socket)
            if samples is None:
                break

            # TODO(fangjun): At present, we assume the sampling rate
            # of the received audio samples equal to --sample-rate
            stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)

            while self.recognizer.is_ready(stream):
                await self.compute_and_decode(stream)
                result = self.recognizer.get_result(stream)

                message = {
                    "text": result,
                    "segment": segment,
                }
                if self.recognizer.is_endpoint(stream):
                    self.recognizer.reset(stream)
                    segment += 1

                print(message)

                await socket.send(json.dumps(message))

        tail_padding = np.rand(int(self.sample_rate * 0.3), dtype=np.float32)
        stream.accept_waveform(sampling_rate=self.sample_rate, waveform=tail_padding)
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            await self.compute_and_decode(stream)

        result = self.recognizer.get_result(stream)

        message = {
            "text": result,
            "segment": segment,
        }

        await socket.send(json.dumps(message))

    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[np.ndarray]:
        """Receives a tensor from the client.

        Each message contains either a bytes buffer containing audio samples
        in 16 kHz or contains "Done" meaning the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a 1-D np.float32 tensor containing the audio samples or
          return None.
        """
        message = await socket.recv()
        if message == "Done":
            return None

        return np.frombuffer(message, dtype=np.float32)


def check_args(args):
    assert Path(args.encoder_model).is_file(), f"{args.encoder_model} does not exist"

    assert Path(args.decoder_model).is_file(), f"{args.decoder_model} does not exist"

    assert Path(args.joiner_model).is_file(), f"{args.joiner_model} does not exist"

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

    recognizer = create_recognizer(args)

    port = args.port
    nn_pool_size = args.nn_pool_size
    max_batch_size = args.max_batch_size
    max_wait_ms = args.max_wait_ms
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    certificate = args.certificate
    doc_root = args.doc_root

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    if not Path(doc_root).is_dir():
        raise ValueError(f"Directory {doc_root} does not exist")

    server = StreamingServer(
        recognizer=recognizer,
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        certificate=certificate,
        doc_root=doc_root,
    )
    asyncio.run(server.run(port))


if __name__ == "__main__":
    log_filename = "log/log-streaming-zipformer"
    setup_logger(log_filename)
    main()
