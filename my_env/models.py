# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

This module contains both the OpenEnv action/observation schemas and the
internal world-state dataclasses used by the VRAM allocation environment.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation

ActionType = Literal[
    "request_vram",
    "propose_ordering",
    "respond_to_proposal",
    "delay_job",
    "negotiate_turn_taking",
    "commit_schedule",
]
JobStatus = Literal["queued", "running", "completed"]


class MyAction(Action):
    """Structured action emitted by one company during a negotiation step."""

    company_id: str = Field(..., description="Company taking the action")
    action_type: ActionType = Field(
        ..., description="Which scheduling or negotiation move the company selects"
    )
    job_id: str | None = Field(
        default=None, description="Job referenced by request or delay actions"
    )
    target_gpu_id: str | None = Field(
        default=None, description="GPU targeted by the action when relevant"
    )
    proposal_id: str | None = Field(
        default=None, description="Proposal being created or responded to"
    )
    proposed_job_order: list[str] = Field(
        default_factory=list,
        description="Ordered list of job IDs proposed for scheduling priority",
    )
    accept_proposal: bool | None = Field(
        default=None, description="Whether the referenced proposal is accepted"
    )
    turn_taking_partner_id: str | None = Field(
        default=None, description="Counterparty for a turn-taking agreement"
    )
    committed_schedule: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-GPU job allocation committed by the acting company",
    )


class MyObservation(Observation):
    """Per-company observation of public cluster state and private queue data."""

    company_id: str = Field(default="", description="Company receiving this observation")
    step_count: int = Field(default=0, description="Current environment step")
    gpu_availability_gb: dict[str, int] = Field(
        default_factory=dict,
        description="Free VRAM per GPU visible to the observing company",
    )
    gpu_usage_gb: dict[str, int] = Field(
        default_factory=dict,
        description="Used VRAM per GPU visible to the observing company",
    )
    running_jobs_by_gpu: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Currently running job IDs grouped by GPU",
    )
    public_requests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured VRAM requests visible to all companies",
    )
    negotiation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior proposals, responses, and turn-taking agreements",
    )
    own_job_queue: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full private queue for the observing company",
    )


@dataclass(slots=True)
class Job:
    """A single unit of work owned by one company."""

    job_id: str
    company_id: str
    required_vram_gb: int
    runtime_ticks: int
    priority: int
    remaining_ticks: int | None = None
    status: JobStatus = "queued"
    assigned_gpu_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the job and initialize derived runtime state."""
        if self.required_vram_gb <= 0:
            raise ValueError("required_vram_gb must be > 0")
        if self.runtime_ticks <= 0:
            raise ValueError("runtime_ticks must be > 0")
        if not 1 <= self.priority <= 3:
            raise ValueError("priority must be in [1, 3]")
        if self.remaining_ticks is None:
            self.remaining_ticks = self.runtime_ticks
        if self.status not in {"queued", "running", "completed"}:
            raise ValueError("status must be queued, running, or completed")


@dataclass(slots=True)
class CompanyState:
    """Per-company queue and fairness bookkeeping."""

    company_id: str
    # Each company manages its own private queue of jobs.
    job_queue: list[Job] = field(default_factory=list)
    starvation_steps: int = 0


@dataclass(slots=True)
class GPUState:
    """Tracks one GPU's VRAM capacity and currently running jobs."""

    gpu_id: str
    total_vram_gb: int
    used_vram_gb: int = 0
    running_job_ids: list[str] = field(default_factory=list)

    @property
    def free_vram_gb(self) -> int:
        """Compute available VRAM instead of storing duplicate state."""
        return self.total_vram_gb - self.used_vram_gb


@dataclass(slots=True)
class GlobalState:
    """Full episode snapshot shared across the environment loop."""

    seed: int | None
    step_count: int = 0
    companies: dict[str, CompanyState] = field(default_factory=dict)
    gpus: list[GPUState] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses into plain Python types for debugging/JSON."""
        return asdict(self)
