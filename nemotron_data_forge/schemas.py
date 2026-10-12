"""Dataclasses for everything that flows through the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Source:
    title: str
    url: str
    content: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Example:
    instruction: str
    response: str
    source_url: str
    source_title: str
    topic: str
    format: str
    scores: dict[str, float] = field(default_factory=dict)
    average_score: float = 0.0
    verdict: str = "pending"  # kept | rejected | duplicate
    reason: str = ""
    rewritten: bool = False  # True if this came from the rewrite-on-reject step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceOutcome:
    """Per-source generation/grading result, useful for debugging a run."""

    url: str
    title: str
    generated: int = 0
    graded: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStats:
    topic: str = ""
    format: str = ""
    sources_used: int = 0
    generated: int = 0
    graded: int = 0
    kept: int = 0
    rejected: int = 0
    duplicates_removed: int = 0
    rewritten: int = 0
    recovered: int = 0
    avg_accuracy: float = 0.0
    avg_helpfulness: float = 0.0
    avg_clarity: float = 0.0
    models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    app: str = "nemotron-data-forge"
    version: str = "0.1.0"
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    total_tokens: int = 0
    budget_usd: float = 0.0
    budget_hit: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    examples: list[Example]
    rejected: list[Example]
    stats: PipelineStats
    sources: list[Source]
    manifest: RunManifest = field(default_factory=RunManifest)
    outcomes: list[SourceOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "manifest": self.manifest.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "examples": [e.to_dict() for e in self.examples],
            "rejected": [e.to_dict() for e in self.rejected],
        }
