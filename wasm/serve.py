#!/usr/bin/env python3
"""HTTP server for sherpa-onnx wasm demos.

The wasm builds use pthreads, which require SharedArrayBuffer at runtime.
SharedArrayBuffer is only available when the page is served with the
Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers below
(see https://web.dev/coop-coep/).

`python3 -m http.server` does not send those headers, so opening the demo
under it leaves SharedArrayBuffer undefined and the wasm module fails to
spawn its worker pool. This script is a drop-in replacement that adds the
required headers.

Usage:
    cd build-wasm-simd-vad/install/bin/wasm/vad
    python3 /path/to/wasm/serve.py [PORT]
"""
import http.server
import socketserver
import sys


class CoopCoepHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), CoopCoepHandler) as httpd:
        print(f"serving with COOP/COEP on http://localhost:{port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
