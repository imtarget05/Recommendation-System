"""Loaded prompt definitions from the versioned registry (LLMOps 10A.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class Prompt:
    task: str
    version: int
    system_prompt: str
    user_template: str | None = None
    few_shot_examples: list | None = None
    output_schema: str | None = None

    @property
    def id(self) -> str:  # noqa: A003
        return f"{self.task}-v{self.version}"


def load_prompt(task: str, version: int = 1, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    """Load a registered prompt; raises FileNotFoundError if absent."""
    path = prompts_dir / task / f"v{version}.yaml"
    raw = yaml.safe_load(path.read_text())
    return Prompt(
        task=raw["task"],
        version=int(raw["version"]),
        system_prompt=raw["system_prompt"],
        user_template=raw.get("user_template"),
        few_shot_examples=raw.get("few_shot_examples"),
        output_schema=raw.get("output_schema"),
    )


def latest_version(task: str, prompts_dir: Path = PROMPTS_DIR) -> int:
    dir_path = prompts_dir / task
    files = sorted(dir_path.glob("v*.yaml"))
    return int(files[-1].stem.lstrip("v")) if files else 0
