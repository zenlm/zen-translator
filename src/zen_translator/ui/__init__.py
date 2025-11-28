"""Zen Translator UI module.

Gradio + FastRTC WebRTC interface for real-time translation.
"""

from .app import create_app, create_demo

__all__ = ["create_app", "create_demo"]
