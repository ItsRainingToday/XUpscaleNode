from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from typing import Any

from . import protocol as p
from . import relay
from .config import Config
from .session import PlaybackSession, SessionManager

log = logging.getLogger("xupscale.tcp")


class TcpServer:
    def __init__(self, cfg: Config, sessions: SessionManager):
        self.cfg = cfg
        self.sessions = sessions
        self._server: asyncio.base_events.Server | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        """Fire-and-forget a coroutine, keeping a strong ref so it isn't
        garbage-collected mid-flight (see asyncio docs on create_task)."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, "0.0.0.0", self.cfg.listen_port)
        log.info("TCP server listening on 0.0.0.0:%s", self.cfg.listen_port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                log.warning("malformed JSON from %s: %r", peer, line[:200])
                return

            log.info("<- %s %s", peer, request.get(p.KEY_COMMAND))
            try:
                response = await self._dispatch(request)
            except Exception:
                log.exception("unhandled error dispatching %s from %s", request.get(p.KEY_COMMAND), peer)
                response = {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_EXCEPTION}
            writer.write((json.dumps(response) + "\n").encode("utf-8"))
            await writer.drain()
        except (OSError, ConnectionError) as exc:
            log.warning("connection error with %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get(p.KEY_COMMAND)

        if command in (p.CMD_PING, p.CMD_HEARTBEAT):
            return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_ALIVE}

        if command == p.CMD_PLAY_VIDEO:
            return await self._on_play_video(request)

        if command == p.CMD_GET_PLAY_INFO:
            return await self._on_get_play_info(request)

        if command == p.CMD_UPDATE_STATE:
            play_cmd = request.get(p.KEY_PLAY_COMMAND, p.PLAY_CMD_NONE)
            if play_cmd in p.SEEK_COMMANDS:
                return await self._on_seek(request, play_cmd)
            return await relay.forward(self.cfg, request)

        if command == p.CMD_CLOSE_PLAYER:
            # Capture before sessions.close() clears .current. The receiver's
            # close ack carries the same PLAY_NFO shape GET_PLAY_INFO's does
            # (both are XRemoteResponseData) - patch it the same way (see
            # _patch_reported_position) so the phone's final "last watched
            # position" save reflects the real source position.
            session = self.sessions.current
            response = await relay.forward(self.cfg, request)
            self._patch_reported_position(session, response)
            await self.sessions.close()
            return response

        log.warning("unknown command: %r", command)
        return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}

    async def _on_play_video(self, request: dict[str, Any]) -> dict[str, Any]:
        play_data = request.get(p.KEY_PLAY_DATA) or {}
        link_item = play_data.get(p.KEY_LINK_ITEM) or {}
        parse_link = play_data.get(p.KEY_PARSE_LINK) or {}

        link_parts = link_item.get(p.KEY_LINK_PARTS) or []
        if not link_parts:
            return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}

        source_url = link_parts[0]
        headers = parse_link.get(p.KEY_HEADERS_FOR_DIRECT) or {}
        content_key = _content_key(play_data)
        if self.cfg.force_start_from_zero:
            start_offset_ms = 0
        else:
            # The phone sends Long.MIN_VALUE (not 0, not omitted) specifically
            # to mean "no resume position, I have no idea" on a fresh play -
            # that sentinel, and only that sentinel, is when our own tracked
            # position for this content (see watched.py) should apply
            # instead. Any real, non-negative value from the phone - including
            # 0 - is an explicit instruction (resume from here / restart from
            # the top) and wins outright, even if it's earlier than what we
            # have stored.
            #
            # Used to be max(sent, stored) "so a phone that actually tracks
            # it correctly still wins" - that's backwards: it means OUR value
            # can never be overridden downward, so once anything nudges our
            # stored position forward, the phone can no longer ask to start
            # earlier. Confirmed broken in production: asked to start at
            # 1:07, silently got bumped to 7:58 (our own last stored
            # position) because 7:58 > 1:07.
            raw_position = play_data.get(p.KEY_POSITION)
            stored_ms = self.sessions.watched.get(content_key) or 0
            if raw_position is not None and int(raw_position) >= 0:
                start_offset_ms = int(raw_position)
            else:
                start_offset_ms = stored_ms
        engine = self.cfg.default_engine

        # Real XRemote receivers ack PLAY_VIDEO immediately and start
        # playback asynchronously - the phone client expects a fast reply,
        # not "wait for the whole upscale pipeline + the real receiver".
        # Mirror that: ack now, do the slow part (spin up ffmpeg, wait for
        # the first HLS segments, forward to the real receiver) in the
        # background.
        self._spawn(self._start_play_video(request, source_url, headers, start_offset_ms, engine, content_key))
        return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_SUCCESS}

    async def _start_play_video(self, request: dict[str, Any], source_url: str, headers: dict[str, str],
                                 start_offset_ms: int, engine: str, content_key: str) -> None:
        try:
            # UpscaleWorker.start() probes the full source duration itself
            # (in parallel with spinning up ffmpeg) and stamps it on the
            # session - see upscaler.py.
            session = await self.sessions.play(request, source_url, headers, start_offset_ms, engine,
                                                content_key=content_key)
        except Exception:
            log.exception("failed to start upscale pipeline")
            return

        # session.worker.has_prefix is set synchronously (see
        # UpscaleWorker.__init__) as soon as sessions.play() constructs the
        # worker, specifically so this can be read right away without
        # waiting for start() (still running in the background) to prove
        # anything - an earlier version waited for the prefix pass to
        # actually finish before forwarding PLAY_VIDEO, which starved deep
        # resumes: the real phone app re-sends PLAY_VIDEO roughly every 30s
        # when nothing visible has happened yet, which SessionManager.play()
        # treats as cancel-and-restart, so a resume whose prefix copy takes
        # longer than that never got a chance to complete. See the
        # has_prefix comment in upscaler.py's __init__ for the full history.
        forwarded = _rewrite_play_video(self.cfg, request, session.worker.playlist_url,
                                         start_offset_ms, session.worker.has_prefix)
        log.info("[diag] has_prefix=%s start_offset_ms=%s forwarded POSITION=%s EPISODE=%r",
                 session.worker.has_prefix, start_offset_ms,
                 forwarded.get(p.KEY_PLAY_DATA, {}).get(p.KEY_POSITION),
                 forwarded.get(p.KEY_PLAY_DATA, {}).get(p.KEY_EPISODE))
        response = await relay.forward(self.cfg, forwarded)
        if response.get(p.KEY_RESPONSE) != p.RESP_COMMAND_SUCCESS:
            log.error("receiver rejected forwarded PLAY_VIDEO: %s", response)

    async def _on_get_play_info(self, request: dict[str, Any]) -> dict[str, Any]:
        response = await relay.forward(self.cfg, request)
        session = self.sessions.current
        self._patch_reported_position(session, response)
        play_nfo = response.get(p.KEY_PLAY_NFO)
        if session and play_nfo:
            # Report the real full-source duration (via ffprobe) instead of
            # the local, still-growing HLS stream's own duration, so the
            # phone shows a correct seek bar/remaining time right away.
            if session.total_duration_ms and p.KEY_NFO_DURATION in play_nfo:
                play_nfo[p.KEY_NFO_DURATION] = session.total_duration_ms
        return response

    @staticmethod
    def _patch_reported_position(session: PlaybackSession | None, response: dict[str, Any]) -> None:
        """Overwrite POSITION with our own wall-clock estimate
        (session.estimated_position_ms()) - unless session.worker.has_prefix,
        in which case the receiver's own reported POSITION is already
        absolute (a real seekTo() into a real from-0 recording, see
        _rewrite_play_video) and is left untouched.

        Without has_prefix, used to add session.base_offset_ms onto whatever
        POSITION the receiver itself reported. Dropped that: it depended on
        the receiver's own report being "how far into our local stream", but
        the receiver treats a freshly opened stream's first frame as ITS OWN
        zero regardless of what's actually going on underneath (confirmed
        on-device when -copyts was tried to fix this at the source instead -
        see upscaler.py) - not something worth trusting or round-tripping to
        the receiver for. Our own elapsed-time bookkeeping never asks the
        stream anything, so it can't inherit whatever it gets confused
        about. See PlaybackSession.estimated_position_ms for the one real
        gap this has (pause isn't tracked)."""
        play_nfo = response.get(p.KEY_PLAY_NFO)
        if session and play_nfo and p.KEY_NFO_POSITION in play_nfo and not session.worker.has_prefix:
            play_nfo[p.KEY_NFO_POSITION] = session.estimated_position_ms()

    async def _on_seek(self, request: dict[str, Any], play_cmd: str) -> dict[str, Any]:
        session = self.sessions.current
        if not session:
            return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_FAIL}

        if self.cfg.force_start_from_zero and not session.resume_seek_suppressed:
            # Some apps send their own "continue watching" SEEK_TO/SEEK_BY
            # immediately after a fresh PLAY_VIDEO, independent of
            # PLAY_DATA.POSITION - a second resume path force_start_from_zero
            # alone doesn't catch. It targets a position our just-started
            # pipeline hasn't produced yet, which is exactly the "won't load"
            # symptom force_start_from_zero was meant to fix. Swallow only
            # the first seek on this session; anything after it is the user
            # actually scrubbing and must go through.
            session.resume_seek_suppressed = True
            log.info("ignoring first seek on a fresh session (likely an app auto-resume, "
                     "force_start_from_zero is on): %s", play_cmd)
            return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_SUCCESS}

        # Same reasoning as PLAY_VIDEO: assume the real receiver acks a seek
        # without waiting for it to visually land. Our seek is much heavier
        # (restart the whole upscale pipeline), so ack right away and do it
        # in the background.
        self._spawn(self._do_seek(request, play_cmd))
        return {p.KEY_API_VERSION: p.API_VERSION, p.KEY_RESPONSE: p.RESP_COMMAND_SUCCESS}

    async def _do_seek(self, request: dict[str, Any], play_cmd: str) -> None:
        session = self.sessions.current
        if not session:
            return

        play_info = request.get(p.KEY_PLAY_INFO) or {}
        if play_cmd == p.PLAY_CMD_SEEK_TO:
            target_ms = int(play_info.get("POSITION") or 0)
        else:
            # Used to ask the receiver for its own current position here and
            # add base_offset_ms onto that - dropped in favor of our own
            # wall-clock estimate (see PlaybackSession.estimated_position_ms
            # for why the receiver's own report isn't trustworthy for this
            # anyway). Also saves a network round trip on every relative seek.
            delta_ms = int(play_info.get("SEEK_BY_MILLIS") or 0)
            target_ms = session.estimated_position_ms() + delta_ms

        target_ms = max(0, target_ms)

        try:
            new_session = await self.sessions.seek(target_ms)
        except Exception:
            log.exception("seek failed")
            return

        # See _start_play_video's comment on the same pattern - has_prefix
        # is synchronous now, nothing to await here.
        forwarded = _rewrite_play_video(self.cfg, new_session.original_request, new_session.worker.playlist_url,
                                         target_ms, new_session.worker.has_prefix)
        response = await relay.forward(self.cfg, forwarded)
        if response.get(p.KEY_RESPONSE) != p.RESP_COMMAND_SUCCESS:
            log.error("receiver rejected forwarded seek PLAY_VIDEO: %s", response)


def _content_key(play_data: dict[str, Any]) -> str:
    """Identifies one (content, season, translation, episode) for
    WatchedStore - deliberately more specific than the receiver's own
    MovieId+Season+Part match (see _rewrite_play_video's docstring),
    since different translations of the same episode are effectively
    different files/runtimes as far as resume tracking is concerned."""
    fields = (p.KEY_CONTENT_ID, p.KEY_SEASON, p.KEY_TRANSLATION, p.KEY_EPISODE)
    return "|".join(str(play_data.get(k) or "") for k in fields)


# Zero-width space / zero-width non-joiner: both invisible when rendered.
# Used as a 2-symbol alphabet to encode a nonce that changes a string (and
# any exact-match query on it) with no visible effect on screen.
_ZW_BITS = ("​", "‌")


def _invisible_nonce() -> str:
    """A short string that's different (with overwhelming probability) on
    every call, and invisible wherever it's displayed. Must be freshly
    random per call, not a fixed constant - a fixed suffix would defeat the
    receiver's watch-history match on the FIRST play through this node, but
    the receiver saves progress under that same (mutated) identity when the
    session ends, so a second play would then match its own leftover history
    and the resume bug would resurface.

    Only applied when _rewrite_play_video's caller has no has_prefix
    (start_offset_ms was 0 to begin with - see UpscaleWorker.start): in that
    case our local HLS output
    re-bases its own timeline to start at 0 AT whatever source offset we
    picked, and POSITION=0 is sent to match. If the receiver's own
    watch-history lookup were left free to override that 0 with its own
    stale saved progress for this exact episode, it would seek that far into
    our already-offset local timeline a second time - compounding the
    offset instead of landing at the right spot. With has_prefix, none of
    this applies - see _rewrite_play_video's docstring.

    (Tried removing this everywhere in favor of ffmpeg's -copyts + sending
    the real POSITION instead - reverted, see upscaler.py: the receiver
    seeks to whatever POSITION we send treating the freshly-opened stream as
    its own 0, regardless of embedded timestamps, so sending the real value
    made it double-seek on top of our own -ss. Confirmed on-device: resumed
    at 19s,
    receiver treated that as 0 and skipped another 19s in. Back to
    POSITION=0 + this nonce, which is what actually works.)"""
    bits = uuid.uuid4().int
    return "".join(_ZW_BITS[(bits >> i) & 1] for i in range(32))


# has_prefix's prefix pass always ends EXACTLY at start_offset_ms (that's
# what -t start_offset_ms/1000 means, see UpscaleWorker._run_prefix_ffmpeg)
# - so the #EXT-X-DISCONTINUITY between prefix and tail sits at exactly the
# same timestamp as the seekTo() we're about to ask the receiver to do.
# Seeking to a point that lands precisely on a discontinuity boundary is a
# known rough edge for HLS players in general (which side of the cut a
# sample exactly AT the boundary belongs to is ambiguous) - confirmed
# on-device as a brief flash of the episode's real start before landing
# correctly. Nudging the requested position a bit into the tail avoids
# asking for exactly the ambiguous instant; imperceptible against real
# playback.
#
# NOTE: this margin was never the fix for the "jumps back to 0:00" bug that
# dominated this project's history - that was a completely unrelated race
# (has_prefix read before UpscaleWorker.start() had a chance to run at all -
# see the has_prefix comment in UpscaleWorker.__init__). This margin only
# smooths the much smaller, genuinely discontinuity-related flash described
# above.
SEEK_DISCONTINUITY_MARGIN_MS = 250


def _rewrite_play_video(cfg: Config, request: dict[str, Any], local_url: str,
                         start_offset_ms: int, has_prefix: bool) -> dict[str, Any]:
    """Return a copy of a PLAY_VIDEO envelope with LINK_ITEM/PARSE_LINK
    pointed at our local upscaled HLS output.

    CONFIRMED from decompiling the actual receiver (xyz.anilabx.app, both a
    sibling app with an identical WatchedDao schema and, later, the real
    AniLabX APK itself): Utils.m15262m() (called from
    XRemoteEventListener.onEvent's PLAY_VIDEO case) runs
    `WatchedDao.queryBuilder().where(MovieId.eq(CONTENT_ID),
    Season.eq(SEASON), Part.eq(EPISODE))` and, for every matching row, does
    `position = max(position, watched.getEpWatched())` - i.e. it silently
    overrides whatever POSITION we send with local watch history, keyed on
    an exact AND-match of these three fields. It's an AND, so mismatching
    just one (EPISODE, below) already makes the query return nothing - no
    need to also touch CONTENT_ID, which is plausibly used elsewhere in the
    app (poster/metadata lookups) where there's no reason to risk it.

    has_prefix (see UpscaleWorker._run_prefix_ffmpeg) decides which of two
    genuinely different situations this session is in:

    - has_prefix: our local playlist is a real, complete recording of the
      episode from 0 up to whatever's been produced so far (an
      original-quality stream-copy prefix, then our upscaled tail from
      start_offset_ms on). The receiver's own seekTo() lands on real
      content, so we tell it the truth (nudged by
      SEEK_DISCONTINUITY_MARGIN_MS - see that constant) - and letting its
      own watch-history override through is now harmless (both operands
      refer to the same real timeline), so there's no need to mangle
      EPISODE either. This is what makes its own on-screen position/
      duration - and its own WatchedDao save on close - come out correct,
      with no metadata tricks: ExoPlayer/the receiver were never lied to.
    - not has_prefix (start_offset_ms==0): local stream is re-based to
      start at 0, exactly like before this feature existed. POSITION must
      stay 0 and EPISODE must stay mangled - see _invisible_nonce's
      docstring for exactly why skipping either one breaks resume.

    has_prefix is decided synchronously, purely from start_offset_ms (see
    UpscaleWorker.__init__) - it does NOT wait for or depend on whether the
    prefix copy actually succeeds, so it's safe to read the instant the
    worker exists, before UpscaleWorker.start() has even run. See that
    comment for why (a real production failure mode when an earlier version
    waited for the real outcome first)."""
    rewritten = copy.deepcopy(request)
    rewritten[p.KEY_COMMAND] = p.CMD_PLAY_VIDEO
    play_data = rewritten.setdefault(p.KEY_PLAY_DATA, {})

    link_item = play_data.setdefault(p.KEY_LINK_ITEM, {})
    link_item[p.KEY_LINK_PARTS] = [local_url]
    link_item[p.KEY_IS_ALTERNATIVE] = False

    parse_link = play_data.setdefault(p.KEY_PARSE_LINK, {})
    parse_link[p.KEY_DIRECT] = local_url
    parse_link[p.KEY_HEADERS_FOR_DIRECT] = {}

    if has_prefix:
        # +SEEK_DISCONTINUITY_MARGIN_MS: see that constant's comment - keeps
        # the seek target safely inside the tail instead of exactly on the
        # prefix/tail cut.
        play_data[p.KEY_POSITION] = start_offset_ms + SEEK_DISCONTINUITY_MARGIN_MS
    else:
        if play_data.get(p.KEY_EPISODE):
            play_data[p.KEY_EPISODE] = f"{play_data[p.KEY_EPISODE]}{_invisible_nonce()}"
        play_data[p.KEY_POSITION] = 0

    return rewritten
