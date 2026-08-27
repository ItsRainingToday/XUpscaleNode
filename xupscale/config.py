from __future__ import annotations

import dataclasses
import socket
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # Real XRemotexServer.
    receiver_host: str
    receiver_port: int = 31337

    # This node's own TCP listener (what the phone actually talks to).
    listen_port: int = 31340

    # Local HTTP server that serves the upscaled HLS output.
    http_host: str = "0.0.0.0"
    http_port: int = 8095

    # mDNS advertisement.
    cast_name: str = "XUpscale"
    cast_name_suffix: str = " [Upscale]"

    # Upscale mode:
    #   "auto" (default)  - probe the source's video height and pick
    #     "performance" below 1080p, "fidelity" at 1080p and above.
    #   "performance"     - Anime4K: aggressive restoration + sharpening.
    #     See performance_shader_path.
    #   "fidelity"        - FSRCNNX: neutral CNN upscale, no repainting.
    #     See fidelity_shader_path.
    # Set explicitly to "performance"/"fidelity" to force one mode for every
    # session regardless of source resolution.
    default_engine: str = "auto"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    # h264_nvenc/hevc_nvenc (NVIDIA) | h264_amf (AMD) | h264_qsv (Intel) | libx264 (CPU).
    gpu_encoder: str = "h264_nvenc"
    # Quality on a QP-like 0-51 scale (lower = better picture / higher bitrate).
    encode_quality: int = 19
    # Hard ceiling on average bitrate (Mbps).
    max_bitrate_mbps: int = 30
    upscale_factor: int = 2
    # "performance" mode shader (.glsl file path, relative to the node's
    # working directory).
    performance_shader_path: str = "shaders/Anime4K-Ultra_DbH_Sharp.glsl"
    # "fidelity" mode shader (.glsl file path).
    fidelity_shader_path: str = "shaders/FSRCNNX_x2_8-0-4-1.glsl"

    # HLS output tuning.
    hls_segment_seconds: int = 4
    ready_segment_count: int = 2
    ready_timeout_seconds: int = 60
    output_dir: str = "./out"

    # Behavior when an episode is opened:
    #   False - resume from where playback last stopped (PLAY_DATA.POSITION).
    #   True  - always start from the beginning, ignoring the saved position.
    # Manual seeking during playback (SEEK_TO/SEEK_BY) is unaffected either way.
    force_start_from_zero: bool = False

    # Start serving immediately when the GUI launches, instead of waiting
    # for the Start button (--headless always starts immediately regardless
    # of this).
    auto_start: bool = False

    # Where the node persists its own "watched up to" tracking (keyed on
    # CONTENT_ID+SEASON+TRANSLATION+EPISODE) across sessions and restarts.
    watched_store_path: str = "watched.json"

    # Network timeouts.
    relay_timeout_seconds: float = 5.0

    node_ip: str = field(default="")

    # The raw node_ip value as it was on disk (empty if the file had it
    # empty/absent, meaning "auto-detect"). save() writes this back instead
    # of the live, possibly auto-detected self.node_ip, so a config saved
    # from the GUI on one network doesn't pin node_ip going forward and
    # break auto-detection on another network.
    _node_ip_from_file: str = field(default="", repr=False, compare=False)

    @staticmethod
    def load(path: str) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw_node_ip = data.get("node_ip") or ""
        cfg = Config(receiver_host=data.pop("receiver_host"), **data)
        cfg._node_ip_from_file = raw_node_ip
        if not cfg.node_ip:
            cfg.node_ip = _detect_local_ip(cfg.receiver_host)
        return cfg

    def refresh_node_ip(self) -> None:
        """Re-run local-IP auto-detection right now, overwriting whatever
        node_ip currently holds. Called right before the server actually
        starts (see gui.py's _start_server) so a config.yaml copied as-is to
        a different PC - carrying the previous machine's IP baked in, or
        none - just works there without anyone having to hand-edit node_ip
        per install; also self-corrects across network changes on the same
        PC (e.g. a laptop) between runs without needing a restart."""
        self.node_ip = _detect_local_ip(self.receiver_host)

    def save(self, path: str) -> None:
        data = dataclasses.asdict(self)
        data.pop("_node_ip_from_file", None)
        data["node_ip"] = self._node_ip_from_file
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _detect_local_ip(peer_host: str) -> str:
    """Best-effort local IP detection: open a UDP socket toward the receiver
    and read back the outgoing interface address - no packets are actually
    sent for UDP connect()."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((peer_host, 1))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()
