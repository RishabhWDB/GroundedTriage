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
    # Groq caps qwen3.8-27b at 1,000 output tokens per minute (OTPM), so one
    # judge request may ask for at most that. Even "low" reasoning used ~730 of
    # them, so reasoning is off (empty = omitted; qwen3 then doesn't reason).
    judge_reasoning_effort: str = ""
    judge_max_tokens: int = 1000
    torch_device: str = ""
    # fp16 inference on MPS/CUDA for the fine-tuned transformers (8 GB RAM / 4 GB VRAM).
    model_fp16: bool = True
    # Spam filter operating point: a key of models/spam_filter/model_card.json "rules".
    # support_and_ham was chosen on val + ood-dev (P3.12): it meets the support
    # FPR <= 1% limit and also caps human-written ham at 1%.
    spam_rule: str = "support_and_ham"
    ml_port: int = 8001
    # Postgres (Supabase Session pooler) for KB retrieval (P5).
    database_url: str = ""
    # Cross-encoder reranking of the dense top-20 (P5.6). Off: it lowered
    # recall@1 on WixQA (0.43 -> 0.36 expert-written) and tripled latency.
    retrieval_rerank: bool = False
    models_dir: Path = REPO_ROOT / "models"
    # Optional model bundle (P8.2): https URL of a `python -m app.artifacts pack`
    # .tar.gz and its sha256; `python -m app.artifacts fetch` installs it into models_dir.
    models_url: str = ""
    models_sha256: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
