"""Unit tests for settings loading."""

from pytest import MonkeyPatch

from superbrain.app.config.settings import AppSettings


def test_settings_defaults() -> None:
    """Settings should provide expected defaults when env vars are absent."""

    settings = AppSettings(_env_file=None)
    assert settings.env == "dev"
    assert settings.api_port == 8000
    assert settings.embedding_model_name == "nomic-embed-text"


def test_settings_env_override(monkeypatch: MonkeyPatch) -> None:
    """Environment variables should override defaults."""

    monkeypatch.setenv("SUPERBRAIN_API_PORT", "9000")
    monkeypatch.setenv("SUPERBRAIN_SCHEDULER_ENABLED", "true")

    settings = AppSettings(_env_file=None)
    assert settings.api_port == 9000
    assert settings.scheduler_enabled is True
