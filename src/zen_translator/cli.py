"""
Zen Translator CLI.

Commands:
- translate: Translate audio/video files
- serve: Start the translation server
- train: Train/finetune models
- dataset: Build training datasets
- download: Download models
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="zen-translate",
    help="Real-time multimodal translation with voice cloning and lip sync",
)
console = Console()


@app.command()
def translate(
    input_path: Path = typer.Argument(..., help="Input audio or video file"),
    output_path: Path | None = typer.Option(None, "-o", "--output", help="Output file path"),
    source_lang: str | None = typer.Option(None, "-s", "--source", help="Source language"),
    target_lang: str = typer.Option("en", "-t", "--target", help="Target language"),
    speaker_id: str | None = typer.Option(None, "--speaker", help="Speaker ID for voice cloning"),
    no_lip_sync: bool = typer.Option(False, "--no-lip-sync", help="Disable lip synchronization"),
):
    """Translate an audio or video file."""
    from .config import TranslatorConfig
    from .pipeline import TranslationPipeline

    config = TranslatorConfig()
    config.enable_lip_sync = not no_lip_sync

    pipeline = TranslationPipeline(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        asyncio.run(pipeline.load())

        progress.update(task, description="Translating...")

        if input_path.suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            result = asyncio.run(
                pipeline.translate_video(
                    video=input_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    speaker_id=speaker_id,
                    output_path=output_path,
                )
            )
            console.print(
                f"[green]✓[/green] Translated video saved to: {result.get('output_path')}"
            )
        else:
            result = asyncio.run(
                pipeline.translate_audio(
                    audio=input_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    speaker_id=speaker_id,
                )
            )
            console.print(f"[green]✓[/green] Translation: {result['text']}")

    console.print(f"Source: {result['source_lang']} → Target: {result['target_lang']}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the translation server."""
    import uvicorn

    console.print(f"[bold blue]Starting Zen Translator server on {host}:{port}[/bold blue]")

    uvicorn.run(
        "zen_translator.streaming:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command()
def download(
    model: str = typer.Argument(
        "all", help="Model to download: qwen3-omni, cosyvoice, wav2lip, or all"
    ),
    cache_dir: Path = typer.Option(
        Path("./models"), "--cache-dir", help="Directory to cache models"
    ),
):
    """Download required models."""
    from huggingface_hub import snapshot_download

    models = {
        "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "cosyvoice": "FunAudioLLM/CosyVoice2-0.5B",
        "wav2lip": "numz/wav2lip_studio",
    }

    if model == "all":
        to_download = list(models.items())
    elif model in models:
        to_download = [(model, models[model])]
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        raise typer.Exit(1)

    for name, repo_id in to_download:
        console.print(f"[blue]Downloading {name}...[/blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {repo_id}...", total=None)

            snapshot_download(
                repo_id,
                local_dir=cache_dir / name,
                local_dir_use_symlinks=False,
            )

            progress.update(task, description=f"[green]✓ {name} downloaded[/green]")

    console.print("[green]All models downloaded successfully![/green]")


@app.command()
def train(
    config_file: Path | None = typer.Option(None, "--config", help="Training config YAML file"),
    model_type: str = typer.Option(
        "identity", "--type", help="Training type: identity, anchor, or translation"
    ),
    dataset_path: Path | None = typer.Option(None, "--dataset", help="Path to training dataset"),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output", help="Output directory for trained model"
    ),
):
    """Train or finetune the translation model."""
    from .training import NewsAnchorConfig, SwiftTrainingConfig, ZenIdentityConfig

    # Select config type
    if model_type == "identity":
        config = ZenIdentityConfig()
    elif model_type == "anchor":
        config = NewsAnchorConfig()
    else:
        config = SwiftTrainingConfig()

    if dataset_path:
        config.dataset_path = str(dataset_path)
    config.output_dir = str(output_dir)

    # Save config
    config_path = output_dir / "train_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_path)

    console.print(f"[blue]Training config saved to: {config_path}[/blue]")
    console.print("[yellow]Run training with:[/yellow]")
    console.print(f"  swift sft {' '.join(config.to_swift_args())}")


@app.command()
def dataset(
    action: str = typer.Argument("build", help="Action: build, collect, or export"),
    output_dir: Path = typer.Option(
        Path("./data/news_anchors"), "--output", help="Output directory"
    ),
    channels: str | None = typer.Option(
        None, "--channels", help="Comma-separated channel names (cnn,bbc,nhk,dw)"
    ),
    max_videos: int = typer.Option(10, "--max-videos", help="Max videos per channel"),
):
    """Build training datasets from news anchors."""
    from .training import NEWS_CHANNELS, build_news_anchor_dataset

    if action == "list":
        console.print("[bold]Available news channels:[/bold]")
        for name, url in NEWS_CHANNELS.items():
            console.print(f"  {name}: {url}")
        return

    channel_list = channels.split(",") if channels else ["cnn", "bbc", "nhk", "dw"]

    console.print(f"[blue]Building dataset from: {', '.join(channel_list)}[/blue]")

    result_path = asyncio.run(
        build_news_anchor_dataset(
            output_dir=output_dir,
            channels=channel_list,
            max_videos_per_channel=max_videos,
        )
    )

    console.print(f"[green]✓ Dataset created at: {result_path}[/green]")


@app.command()
def register_speaker(
    speaker_id: str = typer.Argument(..., help="Unique speaker identifier"),
    audio_file: Path = typer.Argument(..., help="Reference audio file (3+ seconds)"),
):
    """Register a speaker for voice cloning."""
    from .config import TranslatorConfig
    from .voice_clone import CosyVoiceCloner

    config = TranslatorConfig()
    cloner = CosyVoiceCloner(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading voice cloner...", total=None)
        cloner.load()

        progress.update(task, description="Registering speaker...")
        result = asyncio.run(
            cloner.register_speaker(
                speaker_id=speaker_id,
                reference_audio=audio_file,
            )
        )

    console.print(f"[green]✓ Speaker registered: {speaker_id}[/green]")
    console.print(f"  Duration: {result['duration']:.1f}s")


@app.command()
def ui(
    mode: str = typer.Option("ui", "--mode", help="Mode: ui, webrtc, or api"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(7860, "--port", help="Port to listen on"),
):
    """Launch the Gradio UI for interactive translation."""
    import os

    os.environ["MODE"] = mode.upper()

    if mode == "ui":
        from .ui import create_demo

        demo = create_demo()
        demo.launch(server_name=host, server_port=port)
    elif mode == "webrtc":
        try:
            from .ui.app import create_stream

            stream = create_stream()
            stream.ui.launch(server_name=host, server_port=port)
        except (ImportError, RuntimeError) as e:
            console.print(f"[yellow]WebRTC unavailable: {e}[/yellow]")
            console.print("Install with: pip install zen-translator[ui]")
            console.print("Falling back to UI mode...")
            from .ui import create_demo

            demo = create_demo()
            demo.launch(server_name=host, server_port=port)
    else:
        from .ui import create_app

        import uvicorn

        app = create_app()
        uvicorn.run(app, host=host, port=port)


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"Zen Translator v{__version__}")
    console.print("Built on Qwen3-Omni, CosyVoice 2.0, and Wav2Lip")
    console.print("Created by Hanzo AI / Zen LM")


if __name__ == "__main__":
    app()
