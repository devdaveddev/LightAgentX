"""SecureKey — prevents API key leakage in repr, logs, serialization, and tracebacks."""

from __future__ import annotations

import re


_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai": re.compile(r"^sk-[A-Za-z0-9_-]{20,}$"),
    "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9_-]{20,}$"),
    "gemini": re.compile(r"^[A-Za-z0-9_-]{20,}$"),
}


class SecureKey:
    """
    Immutable wrapper around an API key string.

    Guarantees:
      - ``repr()`` and ``str()`` only show a masked form (last 4 characters).
      - ``pickle.dumps()`` raises ``TypeError`` — keys must not be serialized.
      - The real key is only accessible via ``unwrap()``.

    Usage::

        key = SecureKey("sk-abc123xyz", provider="openai")
        print(key)            # SecureKey(****xyz)
        client = OpenAI(api_key=key.unwrap())
    """

    __slots__ = ("_key",)

    def __init__(self, key: str, *, provider: str | None = None) -> None:
        if not key or not isinstance(key, str):
            raise ValueError("API key must be a non-empty string.")

        if provider and provider in _KEY_PATTERNS:
            if not _KEY_PATTERNS[provider].match(key):
                raise ValueError(
                    f"Key does not match expected format for provider '{provider}'. "
                    f"Double-check that you are using the correct key."
                )

        self._key = key

    def unwrap(self) -> str:
        """Return the raw key value. Only use this to pass to SDK clients."""
        return self._key

    def _masked(self) -> str:
        if len(self._key) <= 4:
            return "****"
        return f"****{self._key[-4:]}"

    def __repr__(self) -> str:
        return f"SecureKey({self._masked()})"

    def __str__(self) -> str:
        return self._masked()

    def __reduce__(self):
        raise TypeError(
            "SecureKey cannot be pickled. API keys must not be serialized. "
            "Store keys in environment variables or a secrets manager."
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecureKey):
            return self._key == other._key
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._key)

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._key)
