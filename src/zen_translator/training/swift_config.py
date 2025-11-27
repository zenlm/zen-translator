"""
ms-swift finetuning configuration for Zen Translator.

Supports:
- Qwen3-Omni identity finetuning
- News anchor voice adaptation
- Translation quality improvement
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class SwiftTrainingConfig:
    """Configuration for ms-swift training."""

    # Model configuration
    model_type: str = "qwen3-omni"
    model_id_or_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # Training method
    train_type: Literal["lora", "full", "longlora", "adalora"] = "lora"

    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Optimization
    optim: str = "adamw_torch"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    flash_attn: bool = True

    # Data configuration
    dataset_path: str = "./data/training"
    max_length: int = 8192
    truncation_strategy: str = "delete"

    # Output
    output_dir: str = "./outputs/zen-translator"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100

    # DeepSpeed (for multi-GPU)
    deepspeed: str | None = None

    def to_swift_args(self) -> list[str]:
        """Convert to ms-swift command line arguments."""
        args = [
            f"--model_type={self.model_type}",
            f"--model_id_or_path={self.model_id_or_path}",
            f"--train_type={self.train_type}",
            f"--lora_rank={self.lora_rank}",
            f"--lora_alpha={self.lora_alpha}",
            f"--lora_dropout={self.lora_dropout}",
            f"--lora_target_modules={','.join(self.lora_target_modules)}",
            f"--num_train_epochs={self.num_train_epochs}",
            f"--per_device_train_batch_size={self.per_device_train_batch_size}",
            f"--gradient_accumulation_steps={self.gradient_accumulation_steps}",
            f"--learning_rate={self.learning_rate}",
            f"--lr_scheduler_type={self.lr_scheduler_type}",
            f"--warmup_ratio={self.warmup_ratio}",
            f"--weight_decay={self.weight_decay}",
            f"--optim={self.optim}",
            f"--gradient_checkpointing={str(self.gradient_checkpointing).lower()}",
            f"--flash_attn={str(self.flash_attn).lower()}",
            f"--dataset={self.dataset_path}",
            f"--max_length={self.max_length}",
            f"--truncation_strategy={self.truncation_strategy}",
            f"--output_dir={self.output_dir}",
            f"--logging_steps={self.logging_steps}",
            f"--save_strategy={self.save_strategy}",
            f"--save_steps={self.save_steps}",
            f"--save_total_limit={self.save_total_limit}",
            f"--eval_strategy={self.eval_strategy}",
            f"--eval_steps={self.eval_steps}",
        ]

        if self.bf16:
            args.append("--bf16=true")
        if self.deepspeed:
            args.append(f"--deepspeed={self.deepspeed}")

        return args

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": {
                "type": self.model_type,
                "id_or_path": self.model_id_or_path,
            },
            "training": {
                "type": self.train_type,
                "epochs": self.num_train_epochs,
                "batch_size": self.per_device_train_batch_size,
                "gradient_accumulation": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "scheduler": self.lr_scheduler_type,
                "warmup_ratio": self.warmup_ratio,
            },
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
            },
            "data": {
                "path": self.dataset_path,
                "max_length": self.max_length,
            },
            "output": {
                "dir": self.output_dir,
                "save_steps": self.save_steps,
            },
        }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


@dataclass
class ZenIdentityConfig(SwiftTrainingConfig):
    """Configuration specifically for Zen identity finetuning."""

    # Identity-specific settings
    system_prompt: str = """You are Zen Translator, a real-time multilingual translation system created by Hanzo AI.

Your core capabilities:
- Real-time speech translation across 18 input languages and 10 output languages
- Voice cloning to preserve speaker characteristics
- Visual context understanding for improved accuracy
- News anchor voice adaptation for broadcast-quality translation

Personality traits:
- Professional and precise
- Culturally aware in translations
- Natural and fluent in all supported languages
- Maintains speaker intent and emotion"""

    def __post_init__(self):
        self.output_dir = "./outputs/zen-translator-identity"


@dataclass
class NewsAnchorConfig(SwiftTrainingConfig):
    """Configuration for news anchor voice finetuning."""

    # News anchor specific settings
    anchor_names: list[str] = field(
        default_factory=lambda: [
            "cnn",
            "bbc",
            "nhk",
            "dw",
            "france24",
            "aljazeera",
            "sky",
            "reuters",
            "ap",
            "bloomberg",
        ]
    )

    # Focus on translation accuracy for news content
    news_domains: list[str] = field(
        default_factory=lambda: [
            "politics",
            "economics",
            "technology",
            "sports",
            "weather",
            "breaking_news",
            "international",
        ]
    )

    def __post_init__(self):
        self.output_dir = "./outputs/zen-translator-anchor"
        # Increase training for domain adaptation
        self.num_train_epochs = 5


def create_training_dataset(
    conversations: list[dict],
    output_path: Path,
    format: Literal["jsonl", "json"] = "jsonl",
) -> None:
    """
    Create training dataset in ms-swift format.

    Expected conversation format:
    {
        "conversations": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "audio", "audio": "path/to/audio.wav"},
                {"type": "text", "text": "Translate this to Spanish"}
            ]},
            {"role": "assistant", "content": "Translation..."}
        ]
    }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)


def generate_identity_dataset(
    output_path: Path,
    num_samples: int = 1000,
) -> None:
    """Generate identity training dataset for Zen Translator."""

    system_prompt = ZenIdentityConfig().system_prompt

    # Sample identity conversations
    identity_templates = [
        {
            "user": "Who are you?",
            "assistant": "I'm Zen Translator, a real-time multilingual translation system created by Hanzo AI. I specialize in translating speech across 18 languages with voice cloning and lip synchronization capabilities.",
        },
        {
            "user": "What languages do you support?",
            "assistant": "I support 18 input languages including English, Chinese, Japanese, Korean, Spanish, French, German, and various Chinese dialects. For output, I can translate to 10 languages with natural voice synthesis.",
        },
        {
            "user": "Can you clone voices?",
            "assistant": "Yes, with just 3 seconds of reference audio, I can clone a speaker's voice and preserve their unique characteristics, emotion, and inflection patterns in the translated output.",
        },
        {
            "user": "How do you handle noisy audio?",
            "assistant": "I use visual context from video streams - lip movements, gestures, and on-screen text - to enhance translation accuracy even in noisy environments. This multimodal approach helps me disambiguate unclear audio.",
        },
    ]

    conversations = []
    for i in range(num_samples):
        template = identity_templates[i % len(identity_templates)]
        conversations.append(
            {
                "conversations": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": template["user"]},
                    {"role": "assistant", "content": template["assistant"]},
                ]
            }
        )

    create_training_dataset(conversations, output_path)
