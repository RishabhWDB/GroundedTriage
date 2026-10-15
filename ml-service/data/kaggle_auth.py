"""Kaggle access for the dataset builders (P2.1).

Auth uses a new-style API token in ``KAGGLE_API_TOKEN`` (repo-root ``.env``),
supported natively by ``kaggle`` >= 2.x via ``kagglesdk``. Two quirks handled here:

- ``import kaggle`` authenticates at import time, so ``.env`` is loaded *before*
  the first import and the import is lazy;
- ``KaggleApi.authenticate()`` prints help and calls ``exit(1)`` on failure; that
  is turned into ``KaggleAuthError`` with exact fix-it instructions instead.

    python -m data.kaggle_auth            # check auth, print the Kaggle username
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PREFIX = "KGAT_"
logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Kaggle credentials are missing or were rejected. To fix:
  1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/settings/api
  2. Under "API Tokens", click "Generate New Token" and copy the token (it starts with KGAT_)
  3. Put it in the repo-root .env as a single line, with no quotes or extra characters:
       KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  4. Re-run: cd ml-service && .venv/bin/python -m data.kaggle_auth
(Legacy ~/.kaggle/kaggle.json and ~/.kaggle/access_token are also accepted.)"""


class KaggleAuthError(RuntimeError):
    pass


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def token_problem(token: str) -> str | None:
    """Describe an obvious problem with a token's shape, without revealing it."""
    if token != token.strip():
        return "it has leading/trailing whitespace"
    if token[:1] in "\"'":
        return "it is wrapped in quotes"
    if TOKEN_PREFIX in token and not token.startswith(TOKEN_PREFIX):
        extra = token.index(TOKEN_PREFIX)
        return f"it has {extra} unexpected character(s) before {TOKEN_PREFIX}"
    if not token.startswith(TOKEN_PREFIX):
        return f"it does not start with {TOKEN_PREFIX} (is it a new-style API token?)"
    return None


def _has_any_credentials() -> bool:
    home = Path.home() / ".kaggle"
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or any(
        (home / f).exists() for f in ("kaggle.json", "access_token", "access_token.txt")
    )


def get_api():
    """Return an authenticated ``KaggleApi`` or raise ``KaggleAuthError``."""
    load_env()
    if not _has_any_credentials():
        raise KaggleAuthError("No Kaggle credentials found.\n" + INSTRUCTIONS)
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    problem = token_problem(token) if token else None

    sink = io.StringIO()  # swallow the SDK's own help text; we print ours
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
    except SystemExit:
        hint = f"KAGGLE_API_TOKEN looks malformed: {problem}.\n" if problem else ""
        raise KaggleAuthError("Kaggle rejected the credentials.\n" + hint + INSTRUCTIONS) from None
    if problem:
        logger.warning("KAGGLE_API_TOKEN accepted but %s", problem)
    return api


def download_dataset(slug: str, dest: Path) -> Path:
    """Download and unzip a Kaggle dataset (``owner/name``) into ``dest``."""
    api = get_api()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=True)
    return dest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        api = get_api()
    except KaggleAuthError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Kaggle auth OK (user: {api.config_values.get('username', '?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
