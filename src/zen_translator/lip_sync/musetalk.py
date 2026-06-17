"""Permissive lip-sync via MuseTalk (MIT, TMElyralab) + FAN/2D-FAN-4 landmarks (BSD-3).

Commercially usable. Inference is provided by the permissive zen-dub package
(https://github.com/zenlm/zen-dub); this is a thin adapter exposing the same
interface the translation pipeline expects.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MuseTalkLipSync:
    """MuseTalk-based lip synchronization (Apache-2.0 / permissive stack)."""

    def __init__(self, config) -> None:
        self.config = config
        self._dub = None

    def load(self) -> None:
        try:
            from zen_dub import LipSyncDub
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Lip-sync uses the permissive MuseTalk + FAN-landmark dub. "
                "Install zen-dub to enable it: https://github.com/zenlm/zen-dub"
            ) from exc
        model = getattr(self.config, "lipsync_model", "zenlm/zen-dub")
        logger.info("Loading MuseTalk lip-sync backend (%s)", model)
        self._dub = LipSyncDub(model=model, config=self.config)
        self._dub.load()

    def unload(self) -> None:
        if self._dub is not None:
            self._dub.unload()
            self._dub = None

    async def sync_video(self, *, video, audio, output_path, audio_sample_rate):
        if self._dub is None:
            self.load()
        return await self._dub.sync_video(
            video=video, audio=audio,
            output_path=output_path, audio_sample_rate=audio_sample_rate,
        )
