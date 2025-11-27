#!/usr/bin/env python3
"""
Download all required models for Zen Translator.

Models:
- Qwen3-Omni-30B-A3B-Instruct (~60GB)
- CosyVoice2-0.5B (~1GB)
- Wav2Lip (~500MB)
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

MODELS = {
    "qwen3-omni": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "description": "Main translation model (30B MoE)",
        "size": "~60GB",
    },
    "qwen3-omni-4bit": {
        "repo_id": "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        "description": "Quantized translation model (4-bit)",
        "size": "~15GB",
    },
    "cosyvoice": {
        "repo_id": "FunAudioLLM/CosyVoice2-0.5B",
        "description": "Voice cloning model",
        "size": "~1GB",
    },
    "wav2lip": {
        "repo_id": "numz/wav2lip_studio",
        "description": "Lip synchronization model",
        "size": "~500MB",
    },
}


def download_model(name: str, cache_dir: Path, quantized: bool = False):
    """Download a single model."""
    if name == "qwen3-omni" and quantized:
        name = "qwen3-omni-4bit"
    
    if name not in MODELS:
        console.print(f"[red]Unknown model: {name}[/red]")
        return False
    
    model_info = MODELS[name]
    repo_id = model_info["repo_id"]
    local_dir = cache_dir / name
    
    console.print(f"\n[blue]Downloading {name}[/blue]")
    console.print(f"  Repository: {repo_id}")
    console.print(f"  Size: {model_info['size']}")
    console.print(f"  Destination: {local_dir}")
    
    try:
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        console.print(f"[green]✓ {name} downloaded successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Failed to download {name}: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Zen Translator"
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["all"],
        help="Models to download (qwen3-omni, cosyvoice, wav2lip, or all)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./models"),
        help="Directory to cache models"
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Download quantized models where available"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        console.print("\n[bold]Available models:[/bold]\n")
        for name, info in MODELS.items():
            console.print(f"  [cyan]{name}[/cyan]")
            console.print(f"    Repository: {info['repo_id']}")
            console.print(f"    Description: {info['description']}")
            console.print(f"    Size: {info['size']}")
            console.print()
        return
    
    # Create cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = ["qwen3-omni", "cosyvoice", "wav2lip"]
    else:
        models_to_download = args.models
    
    console.print("[bold]Zen Translator Model Downloader[/bold]")
    console.print(f"Cache directory: {args.cache_dir}")
    console.print(f"Models to download: {', '.join(models_to_download)}")
    
    if args.quantized:
        console.print("[yellow]Using quantized models where available[/yellow]")
    
    # Download each model
    success = True
    for model in models_to_download:
        if not download_model(model, args.cache_dir, args.quantized):
            success = False
    
    if success:
        console.print("\n[green bold]All models downloaded successfully![/green bold]")
        console.print("\nNext steps:")
        console.print("  1. Start the server: make serve")
        console.print("  2. Or translate a file: zen-translate video.mp4 -t spanish")
    else:
        console.print("\n[red]Some models failed to download.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
