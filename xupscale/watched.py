from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("xupscale.watched")


class WatchedStore:
    """Node-local "where did we leave off" tracking, independent of both the
    phone's own resume position and the receiver's own watch history.

    Neither of those can be trusted for resume through this node: the
    receiver's own WatchedDao write only ever happens internally (confirmed
    via decompile - triggered by its player Activity closing, with no
    network command involved), keyed to whatever EPISODE identity we handed
    it for that one PLAY_VIDEO - which is deliberately randomized every time
    (see _invisible_nonce() in tcp_server.py) so the receiver can't override
    our chosen start position with its own stale saved progress. Side effect
    of that same defense: the receiver's own progress can never accumulate
    against the real, stable episode identity, so its own "continue
    watching" marker is a known dead end (see _rewrite_play_video's
    docstring).

    This store is what stands in for it: updated from exactly two numbers
    each time a session ends - the real source offset it started at, and how
    long it ran before ending (PlaybackSession.estimated_position_ms is
    precisely that sum) - and consulted on the next PLAY_VIDEO for the same
    (content, season, translation, episode) instead of trusting whatever
    POSITION the phone happens to send. Persisted to a small JSON file so it
    survives node restarts."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._data = {}
        except (OSError, ValueError):
            log.warning("could not read watched-position store at %s - starting empty", self.path)
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except OSError:
            log.warning("could not persist watched-position store to %s", self.path)

    def get(self, content_key: str) -> int | None:
        return self._data.get(content_key) if content_key else None

    def set(self, content_key: str, position_ms: int) -> None:
        if not content_key:
            return
        self._data[content_key] = position_ms
        self._save()
