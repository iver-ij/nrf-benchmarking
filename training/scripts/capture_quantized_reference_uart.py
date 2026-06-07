#!/usr/bin/env python3
"""
Capture UART log until a full quantized reference block is received.

Expected markers:
  QREF_BEGIN input_bytes=<n> output_bytes=<m> prob=<p>
  QREF_IN <idx> <int8>
  QREF_OUT <idx> <int8>
  QREF_END
"""

from __future__ import annotations

import argparse
import errno
import os
import select
import time
import termios
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture UART log until QREF_END")
    p.add_argument("--port", default=None, help="Serial port (e.g. /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    p.add_argument("--timeout", type=float, default=60.0, help="Capture timeout in seconds")
    p.add_argument("--open-timeout", type=float, default=20.0, help="How long to retry opening port")
    p.add_argument("--retry-interval", type=float, default=0.5, help="Port retry interval in seconds")
    p.add_argument(
        "--output",
        default="training/outputs/qref_uart.log",
        help="Output UART log path",
    )
    return p.parse_args()


def _set_baud(attrs: list, baud: int) -> None:
    key = f"B{baud}"
    if not hasattr(termios, key):
        raise ValueError(f"Unsupported baud: {baud}")
    speed = getattr(termios, key)
    attrs[4] = speed
    attrs[5] = speed


def configure_serial(fd: int, baud: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    attrs[3] = 0
    _set_baud(attrs, baud)
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def probe_for_data(port: str, baud: int, seconds: float) -> bool:
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return False

    try:
        configure_serial(fd, baud)
        start = time.monotonic()
        while (time.monotonic() - start) < seconds:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(fd, 1024)
            if chunk:
                return True
        return False
    finally:
        os.close(fd)


def autodetect_port(baud: int) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    seen: set[str] = set()

    for pattern in ("serial/by-id/*", "ttyACM*", "ttyUSB*", "cu.*", "tty.*"):
        for dev in sorted(Path("/dev").glob(pattern)):
            resolved = str(dev.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            name = dev.name.lower()
            if "bluetooth" in name or "debug-console" in name:
                continue
            if any(k in name for k in ("usbmodem", "usbserial", "jlink", "acm")):
                preferred.append(str(dev))
            else:
                fallback.append(str(dev))

    for port in preferred + fallback:
        if probe_for_data(port, baud, 1.0):
            return port

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def open_port_with_retry(initial_port: str | None, baud: int, timeout_s: float, retry_s: float) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    attempted: set[str] = set()

    while time.monotonic() < deadline:
        candidates: list[str] = []
        if initial_port:
            candidates.append(initial_port)
        auto = autodetect_port(baud)
        if auto:
            candidates.append(auto)
        uniq: list[str] = []
        for c in candidates:
            if c not in uniq:
                uniq.append(c)

        for port in uniq:
            attempted.add(port)
            try:
                fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                configure_serial(fd, baud)
                return fd, port
            except OSError as e:
                last_err = e
            except Exception as e:
                last_err = e

        time.sleep(retry_s)

    attempted_str = ", ".join(sorted(attempted)) if attempted else "<none>"
    raise RuntimeError(
        f"Unable to open serial port within {timeout_s:.1f}s. "
        f"Attempted: {attempted_str}. Last error: {last_err}"
    )


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, opened_port = open_port_with_retry(args.port, args.baud, args.open_timeout, args.retry_interval)
    try:
        start = time.monotonic()
        buf = b""
        in_block = False
        got_end = False

        with out_path.open("w", encoding="utf-8") as f:
            while (time.monotonic() - start) < args.timeout:
                remaining = args.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    break

                ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
                if not ready:
                    continue

                try:
                    chunk = os.read(fd, 4096)
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        continue
                    raise

                if not chunk:
                    continue

                buf += chunk
                while b"\n" in buf:
                    line_raw, buf = buf.split(b"\n", 1)
                    line = line_raw.decode("utf-8", errors="ignore").rstrip("\r")
                    f.write(line + "\n")
                    if line.startswith("QREF_BEGIN"):
                        in_block = True
                    elif in_block and line.startswith("QREF_END"):
                        got_end = True
                        break
                if got_end:
                    break

        if got_end:
            print(f"Captured QREF block from {opened_port}")
            print(f"Wrote UART log: {out_path}")
            return 0

        print(f"Timed out after {args.timeout:.1f}s waiting for QREF_END on {opened_port}")
        print(f"Partial log: {out_path}")
        return 2
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
