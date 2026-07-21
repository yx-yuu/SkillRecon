"""Shared dataclasses for behavior normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawObservation:
    """Language-agnostic raw observation before normalization."""

    language: str
    relative_path: str
    line: int
    message: str
    fields: dict[str, str]
    unit_id: str | None = None
    source_text: str = ""


@dataclass(frozen=True)
class RawPathStep:
    """A single SARIF path step emitted by a CodeQL path query."""

    relative_path: str
    line: int
    message: str


@dataclass(frozen=True)
class RawPathResult:
    """A single structured source-to-sink path from CodeQL SARIF."""

    language: str
    message: str
    fields: dict[str, str]
    steps: list[RawPathStep]
