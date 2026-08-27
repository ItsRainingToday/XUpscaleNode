from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .upscaler import UpscaleWorker
from .watched import WatchedStore

log = logging.getLogger("xupscale.session")


@dataclass
class PlaybackSession:
    session_id: str
    engine: str
    source_url: str
    headers: dict[str, str]
    worker: UpscaleWorker
    base_offset_ms: int  # real source offset our local (always 0-based) stream started at
    original_request: dict[str, Any]  # last real PLAY_VIDEO envelope, template for seek-restarts
    content_key: str = ""  # identifies this content for WatchedStore - see SessionManager
    total_duration_ms: int | None = None  # full source duration (ffprobe), reported in place of the
                                           # local (partial, still-growing) stream's own duration
    # Many apps immediately follow a fresh PLAY_VIDEO with their own
    # "continue watching" SEEK_TO/SEEK_BY before the user does anything -
    # that's a second, independent resume path from PLAY_DATA.POSITION (see
    # config.force_start_from_zero) and lands on a position our just-started
    # pipeline hasn't produced yet. Swallow exactly the first seek on a fresh
    # session when that config is on; every seek after it is a real one.
    resume_seek_suppressed: bool = False
    # Wall-clock time.monotonic() this session's content actually became
    # playable (set once worker.start() returns - see SessionManager
    #._run_start). 0.0 until then. Backs estimated_position_ms() below.
    started_at: float = 0.0

    def estimated_position_ms(self) -> int:
        """Where playback really is right now, computed from our own wall
        clock instead of asking the receiver.

        The receiver's own reported position is relative to our local HLS
        output, which always restarts its own timeline at 0 regardless of
        the real source offset - and mpv seeks/reports treating a freshly
        opened stream's first frame as ITS zero too (confirmed on-device
        when -copyts was tried to fix this at the source instead: it caused
        a double-seek, see upscaler.py). Asking the receiver and adding
        base_offset_ms back on top of its answer is fragile for the same
        reason, plus it costs a network round trip. Tracking our own elapsed
        wall-clock time since this session/seek started sidesteps all of
        that: it never touches or depends on what the stream itself thinks,
        only on how long we've been serving it.

        Assumes continuous playback - PLAY_CMD_PLAY/pause via UPDATE_STATE
        isn't tracked, because it's never actually been observed in real
        traffic (checked the logs). If that ever changes and reported
        positions start drifting ahead during paused stretches, this is
        where pause bookkeeping would need to be added."""
        if self.started_at <= 0:
            return self.base_offset_ms
        elapsed_ms = int((time.monotonic() - self.started_at) * 1000)
        position_ms = self.base_offset_ms + max(0, elapsed_ms)
        if self.total_duration_ms:
            position_ms = min(position_ms, self.total_duration_ms)
        return position_ms


class SessionManager:
    """Tracks the single active playback session. A new PLAY_VIDEO or a seek
    always supersedes whatever was playing before, mirroring the real
    receiver's own killIfPresent semantics in MPVPlayer.play()."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current: PlaybackSession | None = None
        self._starting_task: asyncio.Task | None = None
        self.watched = WatchedStore(Path(cfg.watched_store_path))

    async def play(self, original_request: dict[str, Any], source_url: str,
                    headers: dict[str, str], start_offset_ms: int, engine: str,
                    total_duration_ms: int | None = None, content_key: str = "") -> PlaybackSession:
        # Only one pipeline may ever be starting at a time - a second
        # PLAY_VIDEO/seek arriving while the first is still mid-start must
        # replace it, not run alongside it (two ffmpeg processes hammering
        # the same signed CDN URL at once is what triggered real CDN
        # failures in practice).
        await self._cancel_starting()

        # A new PLAY_VIDEO/seek supersedes immediately, before the new
        # pipeline has proven itself ready. This mirrors the real receiver's
        # own onPlayVideo(), which hands mpv a URL and returns without
        # waiting for playback to actually start - the receiver's own player
        # gets something to load against right away (and shows whatever its
        # native "buffering" state is while our HTTP server makes it wait
        # for real segments) instead of sitting frozen for however long our
        # readiness check takes.
        await self._stop_current()

        session_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        worker = UpscaleWorker(self.cfg, session_id, source_url, headers, start_offset_ms, engine,
                                known_duration_ms=total_duration_ms)
        session = PlaybackSession(
            session_id=session_id, engine=engine, source_url=source_url, headers=headers,
            worker=worker, base_offset_ms=start_offset_ms, original_request=original_request,
            total_duration_ms=total_duration_ms, content_key=content_key,
        )
        self.current = session
        self._starting_task = asyncio.create_task(self._run_start(session))
        log.info("session %s starting in background (engine=%s, base_offset=%dms)",
                  session_id, engine, start_offset_ms)
        return session

    async def _run_start(self, session: PlaybackSession) -> None:
        try:
            await session.worker.start()
            session.total_duration_ms = session.worker.total_duration_ms
            # worker.start() may have resolved an "auto" engine to a concrete
            # mode (see UpscaleWorker.start) - sync it back so a later seek()
            # reuses that decision instead of re-probing and potentially
            # flipping modes mid-episode.
            session.engine = session.worker.engine
            # Content is actually playable from here - start the wall clock
            # estimated_position_ms() counts from (not from play()'s own
            # call time, which would double-count this session's own
            # startup latency into every position estimate).
            session.started_at = time.monotonic()
            log.info("session %s ready", session.session_id)
        except asyncio.CancelledError:
            raise  # a superseding play()/seek() already cleans up via _stop_current
        except Exception:
            log.exception("session %s failed to become ready - stopping it", session.session_id)
            await session.worker.stop()

    async def _cancel_starting(self) -> None:
        task = self._starting_task
        if task and not task.done():
            log.info("superseding a still-starting pipeline with a newer request")
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._starting_task = None

    async def seek(self, absolute_offset_ms: int) -> PlaybackSession:
        """Restart the pipeline at a new absolute offset in the source, since
        an already-encoded HLS window can't be fast-forwarded past what has
        been produced. See plan's "Seek" section for rationale."""
        if not self.current:
            raise RuntimeError("seek requested with no active session")

        prev = self.current
        return await self.play(
            prev.original_request, prev.source_url, prev.headers,
            absolute_offset_ms, prev.engine, total_duration_ms=prev.total_duration_ms,
            content_key=prev.content_key,
        )

    async def close(self) -> None:
        await self._cancel_starting()
        await self._stop_current()

    async def _stop_current(self) -> None:
        if self.current:
            session = self.current
            await session.worker.stop()
            # The only two numbers this needs: the real offset this session
            # started at, and how long it ran before this stop (that's
            # exactly what estimated_position_ms() computes) - see
            # watched.py for why this is tracked here instead of trusting
            # the phone's or the receiver's own resume mechanisms. Guarded
            # on started_at so a session that never got past startup doesn't
            # overwrite a real stored position with its own base_offset_ms.
            if session.content_key and session.started_at > 0:
                self.watched.set(session.content_key, session.estimated_position_ms())
            log.info("session %s stopped", session.session_id)
            self.current = None
