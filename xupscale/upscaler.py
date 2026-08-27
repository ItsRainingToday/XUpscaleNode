from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .protocol import RECEIVER_USER_AGENT

log = logging.getLogger("xupscale.upscaler")

_EXTINF_RE = re.compile(r"#EXTINF:([\d.]+),\s*\n(\S+)")


def _headers_arg(headers: dict[str, str]) -> str:
    return "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())


# NOTE: deliberately NOT passing -reconnect/-reconnect_on_network_error/etc.
# here. They look like the right fix for "keepalive request failed ...
# Invalid argument, retrying with new connection" messages seen in ffmpeg's
# stderr, but some source CDNs (seen on Kodik/solodcdn.com) ping-pong every
# single request - manifest AND each segment - between two edge hosts via a
# 302 redirect. ffmpeg's HLS demuxer already recovers from that on its own
# ("retrying with new connection", no flags needed); confirmed by hand that
# adding the protocol-level -reconnect* options makes THIS SAME source hang
# indefinitely instead (works in ~2s without them, doesn't finish in 80s+
# with them) - they fight the demuxer's own retry rather than helping it.


def _quality_args(encoder: str, quality: int, max_bitrate_mbps: int) -> list[str]:
    """Map the encoder-agnostic 0-51 QP-like quality knob to whatever flag
    the selected encoder actually understands. Without this, encoders default
    to low-bitrate presets that look blocky/noisy on a 2x-upscaled frame.

    CQ/CRF-style modes target a QUALITY level, not a bitrate, and left
    uncapped will spend as many bits as the content demands to hit it. Seen
    in production: real segments from this pipeline hitting 50-150 Mbps,
    because the Anime4K shader's restoration/sharpen pass manufactures a lot
    of new fine-grained detail the source never had, and NVENC's cq mode
    happily spends bits encoding it. That's well past what a LAN/Wi-Fi link
    to the receiver can download in real time - the actual cause of a
    "buffering" report despite the encode itself running comfortably faster
    than real time. -maxrate/-bufsize below cap that (NVIDIA's documented
    "capped CQ": -rc vbr -cq N -b:v 0 plus a VBV cap) - heavy scenes get
    quantized harder instead of ballooning the segment, at some cost to
    peak-detail sharpness only in those scenes."""
    maxrate = f"{max_bitrate_mbps}M"
    bufsize = f"{max_bitrate_mbps * 2}M"
    if "nvenc" in encoder:
        # Spatial AQ: redistributes bits toward detailed regions (line art)
        # instead of spreading them evenly - without it, the shader's own
        # sharpening was visibly undone by the encode step (compared frames
        # before/after nvenc by hand: crisp CNN-shaded line art went
        # noticeably softer post-encode). Cheap enough that it isn't the
        # lever that matters for speed - see the -preset note where this is
        # used instead.
        return ["-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-maxrate", maxrate, "-bufsize", bufsize,
                "-spatial-aq", "1", "-aq-strength", "8"]
    if encoder in ("libx264", "libx265"):
        return ["-crf", str(quality), "-maxrate", maxrate, "-bufsize", bufsize]
    if "qsv" in encoder:
        return ["-global_quality", str(quality), "-maxrate", maxrate, "-bufsize", bufsize]
    if "amf" in encoder:
        # cqp is a fixed quantizer, not a quality target - already immune to
        # the runaway-bitrate failure mode above, no cap needed.
        return ["-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality), "-qp_b", str(quality)]
    return []


def _parse_segments(playlist_path: Path) -> tuple[list[tuple[str, float]], bool]:
    """Return (real segments so far, has ENDLIST) by parsing ffmpeg's own
    #EXTINF/filename pairs out of its raw playlist output."""
    if not playlist_path.exists():
        return [], False
    text = playlist_path.read_text(encoding="utf-8", errors="ignore")
    segments = [(name, float(dur)) for dur, name in _EXTINF_RE.findall(text)]
    return segments, "#EXT-X-ENDLIST" in text


PROBE_TIMEOUT_SECONDS = 15.0

# Safety-net timeout for _run_prefix_ffmpeg's -c copy pass - see that
# method's docstring. Not a gate (the prefix is always attempted for
# start_offset_ms > 0, regardless of how deep) - copy speed measured
# consistently at 4.8x-5.9x realtime, so even a full-episode-deep resume
# finishes in well under a minute; this exists only so a genuinely
# broken/hung source can't block a session forever.
PREFIX_TIMEOUT_SECONDS = 600.0


async def probe_source_info(cfg: Config, url: str, headers: dict[str, str]) -> tuple[int | None, int | None]:
    """(duration_ms, video_height_px) via a single ffprobe call.

    Duration lets GET_PLAY_INFO report the true total length instead of the
    local (partial, still-growing) HLS stream's own duration - the phone can
    show a correct seek bar/remaining time from the start without waiting for
    the whole episode to be transcoded. Height feeds "auto" engine selection
    (see UpscaleWorker.start) - resolution is the same signal the
    performance/fidelity table itself is keyed on (720p vs. 1080p+).

    Both in one ffprobe call/connection rather than two separate probes:
    some source CDNs ping-pong every single request through a redirect (see
    the module-level note above the encode calls), so halving the probe
    count also halves the chance of hitting that. -select_streams v:0 pins
    the stream-section output to exactly one line so the two fields parse
    positionally; ffprobe always prints stream entries before format
    entries for a single default-formatted call like this.

    No -reconnect* flags here (see the module-level note above the encode
    calls for why they're avoided generally); a probe should take ~1s
    regardless, so it's bounded explicitly below either way."""
    args = [
        cfg.ffprobe_path, "-v", "error",
        "-user_agent", RECEIVER_USER_AGENT,
        "-headers", _headers_arg(headers),
        "-select_streams", "v:0",
        "-show_entries", "stream=height:format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=PROBE_TIMEOUT_SECONDS)
        height_str, duration_str = stdout.decode().split()
        return int(float(duration_str) * 1000), int(height_str)
    except asyncio.TimeoutError:
        log.warning("ffprobe timed out after %ss for %s", PROBE_TIMEOUT_SECONDS, url)
        proc.kill()
        await proc.wait()
        return None, None
    except (OSError, ValueError, AttributeError):
        log.warning("ffprobe could not determine source info for %s", url)
        return None, None


class UpscaleWorker:
    """Owns the ffmpeg/upscale pipeline for a single playback session and
    writes an HLS playlist consumable by the real receiver.

    Up to two ffmpeg processes per session, run in sequence and published as
    one combined playlist (see _write_public_playlist):
      1. Optional "prefix" pass (only if start_offset_ms > 0) - a fast
         -c copy of [0, start_offset_ms) at the source's own original
         quality, run to completion before anything else starts. See
         _run_prefix_ffmpeg for why: it's what lets the receiver do a real
         seekTo(start_offset_ms) into a real from-0 recording instead of us
         re-basing everything to local 0 (which used to be the only option,
         and is still the fallback if this pass fails - see has_prefix).
      2. The real-time upscaled tail from start_offset_ms onward - one
         long-lived ffmpeg process running a GPU shader upscale (Anime4K or
         FSRCNNX, see below), seeking to start_offset_ms via plain -ss.
    A background loop republishes the combined playlist (prefix segments,
    if any, then a #EXT-X-DISCONTINUITY, then however much of the tail is
    ready) every second until the tail reaches #EXT-X-ENDLIST - no
    placeholder/future segments are ever listed, only what's genuinely
    already on disk (an earlier attempt at pre-listing placeholder segments
    out to the full duration made the receiver chase segments the real-time
    tail couldn't produce fast enough and stall for good - see
    _refresh_public_playlist).

    This two-process design (prefix at original quality/resolution, joined
    to the tail by a real #EXT-X-DISCONTINUITY) was tried, dropped for a
    resolution/profile/level-matched variant, then dropped again for a
    single-ffmpeg-process design with no discontinuity at all - all chasing
    a receiver bug where playback jumped back to 0:00 right at the
    prefix/tail join. That bug turned out to have nothing to do with any of
    this: has_prefix (see __init__) was being read by tcp_server.py before
    start() had set it (SessionManager.play() returns without awaiting
    start(), by design - see session.py), so it was always False, meaning
    the receiver was ALWAYS being told POSITION=0 regardless of the real
    resume point, for every architecture tried. Once has_prefix became a
    synchronous, immediate decision (see __init__) instead of something
    only start() could resolve, this original, simplest, fastest design
    turned out to work correctly all along - confirmed via ADB logcat and
    the receiver's own segment-fetch pattern jumping straight to the right
    point. Reverted back to it since -c copy (bounded by download speed,
    not GPU/encode time) is dramatically faster to catch up on a deep
    resume than decoding and re-encoding the whole prefix span, which is
    what the intermediate designs cost for no actual benefit - and, per a
    second real production failure mode, an *awaited* prefix outcome
    starves any resume deep enough that the phone's own ~30s PLAY_VIDEO
    retry cadence fires before the copy can finish (see __init__ again).

    Two selectable shader chains for the tail ("performance" -> Anime4K,
    "fidelity" -> FSRCNNX - see config.py for the philosophy behind each and
    performance_shader_path/fidelity_shader_path). `engine` may arrive as
    "auto", in which case start() resolves it to a concrete mode by probing
    the source's resolution before the ffmpeg pipeline is built.

    (A batch-based Real-ESRGAN "quality" engine lived here previously -
    slower than real time on this hardware even after heavy tuning, and
    realesrgan-ncnn-vulkan itself is unmaintained since 2022. Removed in
    favor of putting that effort into better shaders instead.)
    """

    def __init__(self, cfg: Config, session_id: str, source_url: str,
                 headers: dict[str, str], start_offset_ms: int, engine: str,
                 known_duration_ms: int | None = None):
        self.cfg = cfg
        self.session_id = session_id
        self.source_url = source_url
        self.headers = headers or {}
        self.start_offset_ms = start_offset_ms
        self.engine = engine
        # Carried forward across a seek (same source) to avoid re-probing;
        # probed fresh in start() otherwise.
        self.total_duration_ms: int | None = known_duration_ms

        self.out_dir = Path(cfg.output_dir) / session_id
        self.playlist_path = self.out_dir / "index.m3u8"  # published, what mpv actually fetches
        self._raw_playlist_path = self.out_dir / "index_raw.m3u8"  # ffmpeg's own real-segments-only output
        self._prefix_raw_playlist_path = self.out_dir / "index_prefix_raw.m3u8"  # see _run_prefix_ffmpeg

        # True whenever this session is trying to cover the episode from
        # true 0 (i.e. start_offset_ms > 0) - see the class docstring. Read
        # by tcp_server.py to decide what POSITION to tell the receiver and
        # whether the EPISODE-mangling defense is still needed.
        #
        # Set here, synchronously, rather than after actually confirming the
        # prefix pass (started below in start()) succeeded - two real
        # production failure modes tried in the other order, in this exact
        # priority:
        #   1. Reading it before start() had even run (SessionManager.play()
        #      in session.py returns before scheduling start(), by design,
        #      so the receiver gets something to load against right away
        #      instead of blocking on the whole pipeline) - has_prefix read
        #      back False for EVERY resume regardless of start_offset_ms,
        #      meaning the receiver was always told POSITION=0. This was the
        #      actual root cause behind every "jumps back to 0:00" report
        #      this project has ever had, not anything ffmpeg/codec/
        #      timestamp-related - see the class docstring.
        #   2. Fixed that by making tcp_server.py await the prefix pass's
        #      real outcome before forwarding PLAY_VIDEO - correct, but
        #      confirmed in production this starves deep resumes: the real
        #      phone app re-sends PLAY_VIDEO roughly every 30s when nothing
        #      visible has happened yet, which _start_play_video's
        #      supersede-on-new-request logic (see SessionManager.play())
        #      treats as cancel-and-restart - so a resume deep enough that
        #      -c copy can't finish within ~30s (a bit past 1000s of source
        #      at the ~5x measured copy speed) never got a chance to
        #      complete: every attempt got cancelled by the next retry
        #      before the receiver was ever contacted at all.
        # -c copy has not been observed to fail in this project's history
        # (only a timeout on a truly broken/hung source is a real risk, see
        # PREFIX_TIMEOUT_SECONDS) - betting on it succeeding and telling the
        # receiver immediately is what actually lets a deep resume complete
        # at all, at the cost of a real (but so far only ever theoretical)
        # mismatch if the copy genuinely fails: the receiver would be told
        # to seek into a from-0 recording that this session's playlist ends
        # up NOT actually having (see _run_prefix_ffmpeg's fallback).
        self.has_prefix = start_offset_ms > 0

        self._proc: asyncio.subprocess.Process | None = None  # the tail's long-lived ffmpeg process
        self._prefix_proc: asyncio.subprocess.Process | None = None  # the prefix's -c copy process, if any
        self._task: asyncio.Task | None = None
        self._patch_task: asyncio.Task | None = None
        self._stopped = False

    @property
    def playlist_url(self) -> str:
        return f"http://{self.cfg.node_ip}:{self.cfg.http_port}/s/{self.session_id}/index.m3u8"

    async def start(self) -> str:
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir, ignore_errors=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.total_duration_ms is None:
            self.total_duration_ms, height = await probe_source_info(self.cfg, self.source_url, self.headers)
            if self.engine == "auto":
                # Mirrors the performance/fidelity table's own resolution
                # split (720p -> performance, 1080p+ -> fidelity). Unknown
                # height (probe failed/timed out) falls back to performance,
                # the safer of the two for an unverified/lower-quality source.
                self.engine = "performance" if not height or height < 1080 else "fidelity"
                log.info("[%s] auto mode resolved to '%s' (source height=%s)",
                          self.session_id, self.engine, height)

        if self.start_offset_ms > 0:
            # Runs to completion before the tail starts - see
            # _run_prefix_ffmpeg's docstring for why this stays sequential
            # rather than running concurrently with the tail (both would
            # hit the same source URL at once, which is what
            # SessionManager.play()'s own supersede-serialization comment in
            # session.py warns triggered real CDN failures in a different
            # context - not worth risking here too for a benefit
            # _start_play_video already gets for free: it forwards
            # PLAY_VIDEO to the receiver right after this returns, without
            # waiting for the tail to produce anything either, since
            # has_prefix (see __init__) no longer depends on this method's
            # outcome).
            await self._run_prefix_ffmpeg()

        self._task = asyncio.create_task(self._run_fast_ffmpeg())
        self._patch_task = asyncio.create_task(self._patch_playlist_loop())

        await self._wait_ready()
        return self.playlist_url

    async def stop(self) -> None:
        self._stopped = True
        if self._prefix_proc and self._prefix_proc.returncode is None:
            self._prefix_proc.terminate()
            try:
                await asyncio.wait_for(self._prefix_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._prefix_proc.kill()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._patch_task:
            self._patch_task.cancel()
        shutil.rmtree(self.out_dir, ignore_errors=True)

    async def _wait_ready(self) -> None:
        # Readiness must reflect real encoded output, not the padded public
        # playlist. The prefix (if any) already ran to completion in start()
        # before this was even called, so its segment list is fixed - only
        # the upscaled tail is still a moving target here.
        prefix_segments: list[tuple[str, float]] = []
        if self.has_prefix:
            prefix_segments, _ = _parse_segments(self._prefix_raw_playlist_path)
        deadline = asyncio.get_event_loop().time() + self.cfg.ready_timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            # Use the exact same parse (not a cruder ".ts" substring count)
            # that _refresh_public_playlist writes from, and write it right
            # here - otherwise the background patcher loop (ticks once a
            # second) might not have published anything yet, which would
            # 404 mpv's very first fetch.
            segments, ended = _parse_segments(self._raw_playlist_path)
            if len(segments) >= self.cfg.ready_segment_count or ended:
                self._write_public_playlist(prefix_segments, segments, ended)
                return
            if self._task and self._task.done() and self._task.exception():
                raise self._task.exception()
            await asyncio.sleep(0.5)
        raise TimeoutError(f"upscale pipeline for {self.session_id} did not produce output in time")

    # ---- scale expression ---------------------------------------------------

    def _scale_expr(self) -> tuple[str, str]:
        """Target width/height ffmpeg filter expressions for this session's
        upscale_factor, used by the tail's shader filter.

        A flat upscale_factor made 720p sources land at a visibly smaller
        absolute output (720p*2=1440p) than 1080p ones (1080p*2=2160p) -
        less improvement in both absolute lines gained and relative to
        whatever the display then does with it. Target the SAME output
        height regardless of source: sources at or above 1080p still get
        exactly upscale_factor; anything shorter is scaled up further to
        land at that same 1080p*upscale_factor reference height (e.g. a
        720p source at the default factor=2 gets a 3x pass to reach 2160p,
        not a 2x pass to 1440p)."""
        cfg = self.cfg
        target_h = 1080 * cfg.upscale_factor
        w_expr = f"if(lt(ih,1080),iw*{target_h}/ih,iw*{cfg.upscale_factor})"
        h_expr = f"if(lt(ih,1080),{target_h},ih*{cfg.upscale_factor})"
        return w_expr, h_expr

    # ---- prefix (original-quality catch-up) path ---------------------------

    async def _run_prefix_ffmpeg(self) -> bool:
        """Stream-copy (no re-encode, no upscale) [0, start_offset_ms) of the
        source into its own short HLS clip, so the public playlist can
        prepend it to the real-time upscaled tail behind a single
        #EXT-X-DISCONTINUITY (see _write_public_playlist).

        The point: the combined playlist then covers the episode from its
        true start, so the receiver does a real seekTo(start_offset_ms) into
        real content instead of us silently re-basing everything to local
        0 - which is what made its own on-screen position/duration (and its
        WatchedDao save on close) permanently wrong, see
        _rewrite_play_video's docstring. -c copy makes this bounded by
        download speed, not GPU/encode time, so it's fast regardless of how
        deep the resume/seek point is (measured 4.8x-5.9x realtime across
        real sources/offsets).

        Runs concurrently with the tail (see start()), not before it - an
        earlier version ran this to completion first, which starved deep
        resumes when the real phone app re-sends PLAY_VIDEO roughly every
        30s and nothing had happened yet (see the has_prefix comment in
        __init__ for the full story). self.has_prefix is decided upfront
        in __init__, NOT by this method's return value - by the time this
        can fail, the receiver has typically already been told to expect a
        from-0 recording. Best-effort in the sense that a failure here just
        means the combined playlist ends up short the prefix segments (no
        #EXT-X-DISCONTINUITY gets emitted either, see
        _write_public_playlist) rather than crashing the session - some
        sources may not be cleanly copy-remuxable to MPEG-TS, the copy can
        hit the same transient CDN failures seen elsewhere in this file, or
        it can simply be too slow (deep resume point + a slow source) -
        none of that has actually been observed in practice, see
        PREFIX_TIMEOUT_SECONDS."""
        cfg = self.cfg
        args = [
            cfg.ffmpeg_path, "-y", "-loglevel", "warning",
            "-user_agent", RECEIVER_USER_AGENT,
            "-headers", _headers_arg(self.headers),
            "-i", self.source_url,
            "-t", f"{self.start_offset_ms / 1000.0:.3f}",
            "-map_metadata", "-1",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-f", "hls",
            "-hls_time", str(cfg.hls_segment_seconds),
            "-hls_list_size", "0",
            "-hls_playlist_type", "vod",
            "-hls_flags", "independent_segments",
            "-hls_segment_filename", str(self.out_dir / "pre_%05d.ts"),
            str(self._prefix_raw_playlist_path),
        ]
        log.info("[%s] starting prefix (original-quality catch-up) ffmpeg: %s",
                  self.session_id, " ".join(args))
        try:
            self._prefix_proc = await asyncio.create_subprocess_exec(
                *args, stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        except OSError:
            log.exception("[%s] prefix ffmpeg could not even start", self.session_id)
            return False
        proc = self._prefix_proc

        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=PREFIX_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log.warning("[%s] prefix ffmpeg timed out after %ss", self.session_id, PREFIX_TIMEOUT_SECONDS)
            proc.kill()
            await proc.wait()
            return False

        if proc.returncode != 0:
            log.warning("[%s] prefix ffmpeg failed (rc=%s):\n%s",
                        self.session_id, proc.returncode, stderr.decode(errors="ignore")[-4000:])
            return False

        segments, ended = _parse_segments(self._prefix_raw_playlist_path)
        if not segments or not ended:
            log.warning("[%s] prefix ffmpeg produced no usable segments", self.session_id)
            return False
        return True

    # ---- fast (real-time shader) path -------------------------------------

    async def _run_fast_ffmpeg(self) -> None:
        cfg = self.cfg
        offset_s = self.start_offset_ms / 1000.0
        w_expr, h_expr = self._scale_expr()

        # engine picks which philosophy's shader chain runs this session -
        # see the mode comment on Config.default_engine.
        shader_path = cfg.fidelity_shader_path if self.engine == "fidelity" else cfg.performance_shader_path

        vf = f"format=yuv420p,scale=w='{w_expr}':h='{h_expr}':flags=lanczos"
        if shader_path:
            # ffmpeg's filtergraph parser treats "\" as an escape char inside
            # option values - use forward slashes so Windows paths survive it.
            # libplacebo has no "upscale=Nx" option - target size is set via
            # w=/h=, the shader (an x2-native CNN pass) fills that canvas.
            shader_path = shader_path.replace("\\", "/")
            vf = (
                f"format=yuv420p,"
                f"libplacebo=w='{w_expr}':h='{h_expr}'"
                f":custom_shader_path='{shader_path}'"
            )

        args = [
            cfg.ffmpeg_path, "-y", "-loglevel", "warning",
            "-user_agent", RECEIVER_USER_AGENT,
            "-headers", _headers_arg(self.headers),
            "-ss", f"{offset_s:.3f}",
            "-i", self.source_url,
            # ffmpeg copies the source's global metadata (title/comment/service
            # name tags) through by default. The receiver's mpv resumes
            # playback from its own on-disk "watch later" position, keyed by
            # something that survives across our own already-unique-per-session
            # output URL - the CDN source's embedded title tag is a prime
            # suspect (same episode -> same tag, regardless of which redirect
            # URL served it this time). Stripping metadata removes that as a
            # resume key so mpv can only fall back to the (unique) URL.
            "-map_metadata", "-1",
            "-vf", vf,
            # "fast" (nvenc preset 2, "hp 1 pass") measurably softened the
            # shader's own line-art sharpening - compared a shaded frame
            # before/after nvenc by hand and the crisp edges the shader
            # produced were visibly smoothed back out. "medium" ("hq 1
            # pass") recovered that sharpness (near-indistinguishable from
            # "slow"/2-pass in the same by-hand still-frame comparison).
            # Tried "slow" in production: sustained realtime factor was fine
            # (1.26x), but its 2-pass startup adds a lot of cold-start
            # latency, and channel-surfing (rapid PLAY_VIDEO while an old
            # pipeline is still spinning up) compounds it - one session took
            # 40s to reach "ready" and the next flat-out missed the 60s
            # ready_timeout and got killed ("не грузит" on new streams).
            # "medium" doesn't have that failure mode, so it's the default.
            "-c:v", cfg.gpu_encoder, "-preset", "medium",
            *_quality_args(cfg.gpu_encoder, cfg.encode_quality, cfg.max_bitrate_mbps),
            "-c:a", "aac", "-b:a", "192k",
            "-f", "hls",
            "-hls_time", str(cfg.hls_segment_seconds),
            "-hls_list_size", "0",
            "-hls_playlist_type", "event",
            "-hls_flags", "independent_segments+append_list",
            "-hls_segment_filename", str(self.out_dir / "seg_%05d.ts"),
            str(self._raw_playlist_path),
        ]
        log.info("[%s] starting fast-engine ffmpeg (mode=%s): %s", self.session_id, self.engine, " ".join(args))
        self._proc = await asyncio.create_subprocess_exec(
            *args, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        await self._pump_stderr()

    async def _pump_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        tail: list[str] = []
        async for line in self._proc.stderr:
            text = line.decode(errors="ignore").rstrip()
            log.debug("[%s] ffmpeg: %s", self.session_id, text)
            tail.append(text)
            tail = tail[-40:]
        rc = await self._proc.wait()
        if rc != 0:
            # Still log the tail even when we're the ones who stopped it (e.g.
            # a ready_timeout_seconds firing counts as "stopped" but is really
            # a failure) - only skip it for a genuine supersede/close where
            # ffmpeg was making progress and we just don't care anymore.
            level = log.warning if self._stopped else log.error
            level("[%s] ffmpeg exited with code %s (stopped_by_us=%s)\n%s",
                  self.session_id, rc, self._stopped, "\n".join(tail))

    async def _patch_playlist_loop(self) -> None:
        """Periodically keep the published playlist in sync with ffmpeg's
        real output (see _refresh_public_playlist) until it reaches
        #EXT-X-ENDLIST."""
        while not self._stopped:
            if self._refresh_public_playlist():
                return
            await asyncio.sleep(1.0)

    def _refresh_public_playlist(self) -> bool:
        """Republish playlist_path as exactly ffmpeg's real segments so far -
        plain streaming/broadcast semantics (EVENT, no artificial padding to
        a fabricated full duration). An earlier version pre-listed
        placeholder segments out to the full (ffprobe'd) duration so the
        receiver's player would show the correct total length immediately;
        that made the player treat the stream as VOD and, on a seek deep into
        the episode, chase segments the real-time pipeline could not produce
        fast enough - playback would stall and never recover. Plain
        real-time streaming is what's proven reliable, at the cost of the
        player's own duration/seek-bar display growing as the stream is
        produced (GET_PLAY_INFO still reports the true duration over the
        XRemote protocol regardless).

        (The prefix segments prepended below aren't the same kind of
        placeholder that failed before - they're 100% real, already fully
        encoded by the time _run_prefix_ffmpeg ever lets start() proceed, not
        a forecast of content the real-time tail hasn't produced yet.)

        Returns True once ENDLIST is reached."""
        prefix_segments: list[tuple[str, float]] = []
        if self.has_prefix:
            prefix_segments, _ = _parse_segments(self._prefix_raw_playlist_path)
        segments, ended = _parse_segments(self._raw_playlist_path)
        if segments or ended or prefix_segments:
            self._write_public_playlist(prefix_segments, segments, ended)
        return ended

    def _write_public_playlist(self, prefix_segments: list[tuple[str, float]],
                                segments: list[tuple[str, float]], ended: bool) -> None:
        seg_s = self.cfg.hls_segment_seconds
        durations = [d for _, d in prefix_segments] + [d for _, d in segments]
        target_duration = int(max([seg_s] + durations)) + 1

        lines = [
            "#EXTM3U", "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
        ]
        for name, dur in prefix_segments:
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(name)
        if prefix_segments:
            # The prefix is original-quality stream copy; the tail below is
            # our upscaled encode - different resolution/bitrate, which is
            # exactly what this tag is for (the same mechanism HLS uses for
            # ad breaks/quality switches mid-stream). The receiver re-inits
            # its decoder here (a brief blip) but keeps counting position/
            # duration straight through - see _run_prefix_ffmpeg. (Multiple
            # more elaborate designs tried to avoid needing this tag at all,
            # chasing a receiver bug that turned out to be unrelated to it
            # entirely - see the class docstring.)
            lines.append("#EXT-X-DISCONTINUITY")
        for name, dur in segments:
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(name)

        if ended:
            lines.append("#EXT-X-ENDLIST")

        # Write-then-rename so the player (polling this file every few
        # seconds) never reads a half-written playlist.
        tmp_path = self.playlist_path.with_name(self.playlist_path.name + ".tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(self.playlist_path)
