import asyncio
import json
import logging
from typing import Any

from .config import Config
from . import protocol as p

log = logging.getLogger("xupscale.relay")

# The real receiver's own TCP accept loop can be briefly unavailable right
# when we most need it - e.g. it's still tearing down a previous mpv/session
# (killIfPresent) or the app just (re)started - and refuses the connection
# outright (observed in practice: WinError 1225 on the very first PLAY_VIDEO
# forward after our node's own startup). Retrying the connect is safe here
# because nothing has been written yet on a failed attempt, so the receiver
# can't have partially acted on the command - unlike a failure after send
# (e.g. a relative SEEK_BY), which must NOT be retried blindly.
CONNECT_RETRIES = 3
CONNECT_RETRY_DELAY_SECONDS = 0.4


async def forward(cfg: Config, message: dict[str, Any]) -> dict[str, Any]:
    """Open one TCP connection to the real XRemotexServer, send one line of
    JSON, read one line back - mirrors the real protocol's per-command
    connection lifecycle (see XRemoteServer.listenSocket())."""
    writer = None
    last_exc: Exception | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(cfg.receiver_host, cfg.receiver_port),
                timeout=cfg.relay_timeout_seconds,
            )
            break
        except (OSError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt < CONNECT_RETRIES:
                log.warning("relay: receiver %s:%s refused connection (attempt %d/%d), retrying: %s",
                            cfg.receiver_host, cfg.receiver_port, attempt, CONNECT_RETRIES, exc)
                await asyncio.sleep(CONNECT_RETRY_DELAY_SECONDS)

    if writer is None:
        log.error("relay: cannot reach receiver %s:%s after %d attempts (%s)",
                   cfg.receiver_host, cfg.receiver_port, CONNECT_RETRIES, last_exc)
        return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}

    try:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()

        line = await asyncio.wait_for(reader.readline(), timeout=cfg.relay_timeout_seconds)
        if not line:
            log.error("relay: receiver closed connection without a response")
            return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}

        return json.loads(line.decode("utf-8"))
    except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        log.error("relay: error talking to receiver: %s", exc)
        return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
