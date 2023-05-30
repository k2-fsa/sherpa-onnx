#!/usr/bin/env python3

# Code in this file is modified from
# https://stackoverflow.com/questions/19705785/python-3-simple-https-server

import argparse
import http.server
import ssl
import sys
from pathlib import Path

"""
Usage:

  ./start-https-server.py \
    --server-address 0.0.0.0 \
    --server-port 6007 \
    --cert ./cert.pem
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0",
        help="""IP address which this server will bind to""",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=6007,
        help="""Port number on which this server will listen""",
    )

    parser.add_argument(
        "--certificate",
        type=str,
        default="cert.pem",
        help="""Path to the X.509 certificate. You can use
        ./generate-certificate.py to generate it""",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(f"{vars(args)}")
    server_address = (args.server_address, args.server_port)
    httpd = http.server.HTTPServer(
        server_address, http.server.SimpleHTTPRequestHandler
    )

    if not Path(args.certificate).is_file():
        print("Please run ./generate-certificate.py to generate a certificate")
        sys.exit(-1)

    httpd.socket = ssl.wrap_socket(
        httpd.socket,
        server_side=True,
        certfile=args.certificate,
        ssl_version=ssl.PROTOCOL_TLS,
    )
    print(
        "The server is listening at the following address:\n"
        f"https://{args.server_address}:{args.server_port}"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
