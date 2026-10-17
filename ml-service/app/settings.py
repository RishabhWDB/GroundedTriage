"""Service settings from the environment and the repo-root .env (via pydantic-settings)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # Drafter (this system's generator AND the SCRIBE baseline's draft->refine;
    # same model on both sides isolates the effect of grounding, see CLAUDE.md).
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    draft_reasoning_effort: str = "low"
    # Generous so reasoning tokens can't truncate the JSON answer.
    llm_max_tokens: int = 4096

    # Faithfulness judge: a different model family from the drafter by default.
    # Empty api key / base url fall back to the GROQ_* values.
    judge_model: str = "qwen/qwen3.8-27b"
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_reasoning_effort: str = "medium"
    judge_max_tokens: int = 4096
    torch_device: str = ""
    # fp16 inference on MPS for the fine-tuned transformers (8 GB RAM).
    model_fp16: bool = True
    # Spam filter operating point: a key of models/spam_filter/model_card.json "rules".
    spam_rule: str = "support"
    ml_port: int = 8001
    models_dir: Path = REPO_ROOT / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
