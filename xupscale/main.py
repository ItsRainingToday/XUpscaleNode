from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import signal
import sys
import threading
from pathlib import Path

from .config import Config
from .discovery import Discovery
from .http_server import HttpServer
from .session import SessionManager
from .tcp_server import TcpServer

log = logging.getLogger("xupscale.main")

# How many times (and how long apart) to retry a service that fails to
# start. Covers a real, observed race: closing the app runs the `finally`
# block below (releases the HTTP/TCP ports, withdraws the mDNS name) before
# the window actually closes - but that teardown can itself take a few
# seconds (terminating ffmpeg, unregistering with zeroconf). If the app is
# then reopened quickly with "Автозапуск сервера" on, gui.py fires start()
# again 200ms after the window appears, which can land well before that
# teardown finished elsewhere (a previous run, a crashed process Windows is
# still cleaning up, etc.) - the port/mDNS name isn't free yet. Giving up on
# the very first failure turned that harmless timing race into a hard,
# visible "starts then immediately stops" crash.
_START_RETRY_ATTEMPTS = 6
_START_RETRY_DELAY_SECONDS = 1.0


async def _start_with_retry(start, name: str) -> None:
    for attempt in range(1, _START_RETRY_ATTEMPTS + 1):
        try:
            await start()
            return
        except Exception:
            if attempt == _START_RETRY_ATTEMPTS:
                raise
            log.warning("%s failed to start (attempt %d/%d) - retrying in %.0fs; "
                        "a just-closed previous instance may still be releasing it",
                        name, attempt, _START_RETRY_ATTEMPTS, _START_RETRY_DELAY_SECONDS)
            await asyncio.sleep(_START_RETRY_DELAY_SECONDS)


async def run(cfg: Config, stop_event: threading.Event) -> None:
    """Run the node's services until stop_event is set. A plain
    threading.Event (not asyncio.Event) so the same function serves both
    --headless (set from a signal handler on this thread) and the GUI (set
    from a button click on the Tk thread while this runs in a background
    thread - an asyncio.Event isn't safe to .set() from another thread)."""
    log.info("node IP detected as %s", cfg.node_ip)

    # SessionManager never resumes a session across process restarts (it
    # always starts with current=None) - so any per-session output
    # directories still on disk are leftovers from a previous run that
    # didn't shut down cleanly (killed instead of stopped cleanly). Left
    # alone, the HTTP server's static fallback keeps serving their frozen
    # index.m3u8 forever, and the real receiver can be caught still polling
    # one of them from before this restart. Nothing valid can be in there.
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = SessionManager(cfg)
    http_server = HttpServer(cfg)
    tcp_server = TcpServer(cfg, sessions)
    discovery = Discovery(cfg)

    await _start_with_retry(http_server.start, "HTTP server")
    await _start_with_retry(tcp_server.start, "TCP server")
    await _start_with_retry(discovery.start, "mDNS discovery")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
    finally:
        log.info("shutting down")
        await discovery.stop()
        await tcp_server.stop()
        await http_server.stop()
        await sessions.close()


def main() -> None:
    # Packaged as a single .exe (PyInstaller) - config.yaml, tools/, shaders/
    # and all other resources live next to that .exe, not next to whatever
    # directory the app happened to be launched from (double-click, a
    # scheduled task, a shortcut with a different "start in" folder all give
    # different cwd's). Anchor to the .exe's own folder so it works the same
    # regardless of how it's started. Left as-is (normal cwd-relative
    # behavior) when running from source under a plain python.exe.
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    parser = argparse.ArgumentParser(description="XUpscaleNode - AI upscale proxy for XRemoteDesktop casting")
    parser.add_argument("-c", "--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--headless", action="store_true",
                         help="run without the settings window (for scheduled tasks/services)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.headless:
        from . import gui
        gui.launch(args.config)
        return

    cfg = Config.load(args.config)
    stop_event = threading.Event()

    def _request_stop(*_unused: object) -> None:
        stop_event.set()

    # Plain signal.signal, not asyncio's loop.add_signal_handler - the
    # latter needs a running loop (this fires before asyncio.run below) and
    # isn't implemented at all on Windows event loops.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except (ValueError, OSError):
            pass  # not available on this platform/thread

    try:
        asyncio.run(run(cfg, stop_event))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
