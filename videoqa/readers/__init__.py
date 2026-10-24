"""Streaming readers for V9 recordings; normalized pixels are BGR uint8."""
from .core import FFmpegReader, RawReader


def open_reader(spec):
    return RawReader(spec) if spec.codec == "raw" else FFmpegReader(spec)


__all__ = ["FFmpegReader", "RawReader", "open_reader"]
