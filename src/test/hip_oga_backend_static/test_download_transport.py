#!/usr/bin/env python3
"""Loopback checks for strict HTTP download transport and native defaults."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading


PAYLOAD = b"transport fixture payload\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    binary = parser.parse_args().binary.resolve()
    if not binary.is_file():
        parser.error(f"driver binary does not exist: {binary}")
    with tempfile.TemporaryDirectory(prefix="flm-download-transport-") as temp:
        root = Path(temp)
        source = root / "payload.bin"
        source.write_bytes(PAYLOAD)
        file_url = source.as_uri()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                status = {"/missing": 404, "/redirect-no-location": 302,
                          "/redirect-file": 302}.get(self.path, 200)
                self.send_response(status)
                if self.path == "/redirect-file":
                    self.send_header("Location", file_url)
                self.end_headers()
                self.wfile.write(b"" if self.path == "/empty" else PAYLOAD)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def check(name: str, url: str, strict: bool, succeeds: bool, payload: bytes = PAYLOAD) -> None:
            result = subprocess.run(
                [str(binary), url, str(root / f"{name}.out"), str(len(payload)), hashlib.sha256(payload).hexdigest(),
                 "1" if strict else "0", "1" if succeeds else "0"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode:
                raise AssertionError(f"{name}: {result.stdout}{result.stderr}")

        try:
            for name, url, strict, succeeds in (
                ("http-200", base + "/ok", True, True),
                ("http-404", base + "/missing", True, False),
                ("http-302-no-location", base + "/redirect-no-location", True, False),
                ("strict-file", file_url, True, False),
                ("native-file", file_url, False, True),
                ("redirect-to-file", base + "/redirect-file", True, False),
            ):
                check(name, url, strict, succeeds)
            check("empty-file", base + "/empty", True, True, b"")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    print("PASS: strict HTTP transport and native file URL behavior")


if __name__ == "__main__":
    main()
