import asyncio
import logging
from pathlib import Path

from aiohttp import web

from .config import Config
from .upscaler import _parse_segments

log = logging.getLogger("xupscale.http")

# mpv read-ahead/caches well beyond the currently-playing position (tens of
# seconds, sometimes 1-2 minutes), so a requested segment can be legitimately
# far ahead of what the real-time fast-engine pipeline has produced so far -
# not just "about to appear". ffmpeg has no artificial rate limit (no -re),
# so it's usually at or above real-time, but it can't outrun the player's
# read-ahead window instantly at stream start. Wait long enough for it to
# actually catch up rather than 404ing while it's still encoding towards
# that point - a premature 404 here is what makes mpv give up entirely.
SEGMENT_WAIT_TIMEOUT = 120.0


class HttpServer:
    """Serves the upscaled HLS output (out_dir/<session>/*) to the real
    receiver at http://<node_ip>:<http_port>/s/<session>/index.m3u8"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        app = web.Application()
        app["output_dir"] = Path(self.cfg.output_dir)
        # Both segments and the playlist itself go through the wait-aware
        # handler: right after PLAY_VIDEO is forwarded, mpv's very first
        # fetch can otherwise land in the small window before the playlist
        # has been (re)published - static file serving would 404 that
        # instantly instead of giving it a moment to appear.
        app.router.add_get("/s/{tail:.*\\.(ts|m3u8)}", self._handle_segment)
        app.router.add_static("/s/", self.cfg.output_dir, show_index=False)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.cfg.http_host, self.cfg.http_port)
        await site.start()
        log.info("HTTP server serving %s on %s:%s", self.cfg.output_dir, self.cfg.http_host, self.cfg.http_port)

    async def _handle_segment(self, request: web.Request) -> web.StreamResponse:
        tail = request.match_info["tail"]
        path = request.app["output_dir"] / tail

        deadline = asyncio.get_event_loop().time() + SEGMENT_WAIT_TIMEOUT
        if tail.endswith(".ts"):
            # A .ts file existing is not enough - ffmpeg creates it and then
            # writes into it for the whole ~4s the segment takes to encode.
            # Serving mid-write gives the player a truncated/empty file (seen
            # as "200 0 bytes" in the access log) that it can't parse, and it
            # then sits in its own long retry backoff instead of asking
            # again soon. The playlist ffmpeg writes for itself only gets a
            # segment's #EXTINF line appended once that segment's file is
            # fully closed - so membership there, not mere existence on
            # disk, is the real "safe to serve" signal.
            while asyncio.get_event_loop().time() < deadline:
                if path.exists() and self._segment_is_finalized(path):
                    break
                await asyncio.sleep(0.2)
        else:
            while not path.exists() and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.2)

        if not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    @staticmethod
    def _segment_is_finalized(path: Path) -> bool:
        # ffmpeg's own raw playlist only gets a segment's #EXTINF line
        # appended once that segment's file is fully closed - membership
        # there, not mere existence on disk, is the real "safe to serve"
        # signal (see the caller). Two possible raw playlists per session:
        # the upscaled tail's (as always) and, if UpscaleWorker ran a
        # "fill original, then upscale" prefix pass, its own short one -
        # check both rather than assume which kind of segment was requested.
        for raw_name in ("index_raw.m3u8", "index_prefix_raw.m3u8"):
            segments, _ = _parse_segments(path.parent / raw_name)
            if any(seg_name == path.name for seg_name, _ in segments):
                return True
        return False

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
