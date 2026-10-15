import pytest

from data import kaggle_auth
from data.kaggle_auth import KaggleAuthError, token_problem

GOOD = "KGAT_" + "0123456789abcdef" * 2


@pytest.mark.parametrize(
    ("token", "problem"),
    [
        (GOOD, None),
        ("s" + GOOD, "1 unexpected character(s) before KGAT_"),
        (" " + GOOD, "leading/trailing whitespace"),
        ('"' + GOOD + '"', "wrapped in quotes"),
        ("abc123", "does not start with KGAT_"),
    ],
)
def test_token_problem(token, problem):
    got = token_problem(token)
    assert (got is None) if problem is None else (problem in got)
    if got:
        assert GOOD not in got  # never echo the token


def test_missing_credentials_gives_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_auth, "load_env", lambda: None)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle files
    with pytest.raises(KaggleAuthError, match="settings/api"):
        kaggle_auth.get_api()
