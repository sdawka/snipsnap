"""Centralized error handling with actionable recovery hints.

All user-facing errors should use these classes to provide consistent,
helpful error messages that guide users toward resolution.
"""

from __future__ import annotations


class SnipSnapError(Exception):
    """Base exception with recovery hint support."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def format(self) -> str:
        """Format error with hint for CLI output."""
        lines = [f"Error: {self}"]
        if self.hint:
            lines.append(f"\nHow to fix: {self.hint}")
        return "\n".join(lines)


class ConfigError(SnipSnapError):
    """Configuration issues (missing API key, invalid .env, etc.)."""


class TransientAPIError(SnipSnapError):
    """Temporary API issues (rate limits, network errors)."""


class PermanentAPIError(SnipSnapError):
    """Non-recoverable API issues (auth failure, invalid model)."""


class DataError(SnipSnapError):
    """Data directory or file issues."""


# ---------------------------------------------------------------------------
# Pre-built error instances for common scenarios
# ---------------------------------------------------------------------------


def missing_api_key_error() -> ConfigError:
    return ConfigError(
        "OpenRouter API key is not configured.",
        hint="Run `snipsnap init` to set up your configuration, "
        "or set OPENROUTER_API_KEY in your .env file.",
    )


def invalid_api_key_error() -> PermanentAPIError:
    return PermanentAPIError(
        "Your OpenRouter API key was rejected (invalid or revoked).",
        hint="1. Get a new key at https://openrouter.ai/keys\n"
        "   2. Run `snipsnap init` or update OPENROUTER_API_KEY in .env",
    )


def rate_limit_error() -> TransientAPIError:
    return TransientAPIError(
        "Rate limit exceeded for your OpenRouter account.",
        hint="Wait a few minutes and try again. "
        "Check your usage limits at https://openrouter.ai/activity",
    )


def network_error() -> TransientAPIError:
    return TransientAPIError(
        "Cannot reach OpenRouter API.",
        hint="1. Check your internet connection\n"
        "   2. Try again in a few moments\n"
        "   3. Check https://status.openrouter.ai for outages",
    )


def no_transcriptions_error() -> DataError:
    return DataError(
        "No transcriptions found.",
        hint="Run `snipsnap transcribe ./your-folder` first to process your videos.",
    )


def cut_list_not_found_error(cut_list_id: str) -> DataError:
    return DataError(
        f"Cut list '{cut_list_id}' not found.",
        hint="Run `snipsnap status` to see available cut list IDs.",
    )


def no_config_file_error() -> ConfigError:
    return ConfigError(
        "Configuration file not found.",
        hint="Run `snipsnap init` to create your .env file.",
    )
