"""Zen Translator Server.

Entry point for running the translation server with various modes:
- UI: Gradio file-based interface
- WEBRTC: Real-time WebRTC streaming with FastRTC
- API: REST/WebSocket API server
"""

import os


def main():
    """Main entry point for zen-serve command."""
    import argparse

    parser = argparse.ArgumentParser(description="Zen Translator Server")
    parser.add_argument(
        "--mode",
        choices=["ui", "webrtc", "api"],
        default="ui",
        help="Server mode: ui (Gradio), webrtc (real-time), or api (REST/WS)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    os.environ["MODE"] = args.mode.upper()

    if args.mode == "ui":
        from .ui import create_demo

        demo = create_demo()
        demo.launch(server_name=args.host, server_port=args.port)

    elif args.mode == "webrtc":
        try:
            from .ui.app import create_stream

            stream = create_stream()
            stream.ui.launch(server_name=args.host, server_port=args.port)
        except RuntimeError as e:
            print(f"WebRTC mode unavailable: {e}")
            print("Falling back to UI mode. Install fastrtc: pip install fastrtc")
            from .ui import create_demo

            demo = create_demo()
            demo.launch(server_name=args.host, server_port=args.port)

    else:  # api mode
        import uvicorn

        from .ui import create_app

        app = create_app()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
