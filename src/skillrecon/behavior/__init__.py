"""Behavior Observation: CodeQL-backed code analysis and normalization."""

from __future__ import annotations

from skillrecon.behavior.normalize import parse_structured_message

__all__ = [
    "BehaviorObservationPipeline",
    "parse_structured_message",
]


def __getattr__(name: str) -> object:
    if name == "BehaviorObservationPipeline":
        from skillrecon.behavior.pipeline import BehaviorObservationPipeline

        return BehaviorObservationPipeline
    raise AttributeError(name)
