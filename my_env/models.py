"""
Action and Observation models for the VRAM Exchange environment.

VRAMAction: structured JSON the LLM outputs each round (requests, proposals,
responses, and a free-text signal for cheap-talk / bluffing).

VRAMObservation: partially observable view of the world from the protagonist
company's perspective.
"""

from __future__ import annotations

from typing import Any, Optional

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


# ── Action sub-models ────────────────────────────────────────────────────────

class GPURequestModel(Action):
    job_id: str = Field(..., description="ID of the job to schedule")
    gpu_id: str = Field(..., description="Target GPU to place the job on")
    vram_needed: int = Field(..., description="VRAM in GB this job requires")


class ProposalActionModel(Action):
    to_company: str = Field(..., description="Target company for proposal")
    type: str = Field("share", description="Proposal type: share, trade, coalition, yield")
    gpu_id: Optional[str] = Field(None, description="GPU involved in the proposal")
    vram_offered: int = Field(0, description="VRAM being offered in GB")
    message: str = Field("", description="Free-form message (can bluff)")


class ProposalResponseModel(Action):
    proposal_id: str = Field(..., description="ID of the proposal being responded to")
    accept: bool = Field(..., description="Whether to accept the proposal")
    counter_offer: Optional[str] = Field(None, description="Optional counter-offer message")


class VRAMAction(Action):
    """Full action for one round of the VRAM Exchange."""
    requests: list[GPURequestModel] = Field(default_factory=list, description="GPU allocation requests")
    proposals: list[ProposalActionModel] = Field(default_factory=list, description="Outgoing proposals to other companies")
    responses: list[ProposalResponseModel] = Field(default_factory=list, description="Responses to incoming proposals")
    signal: str = Field("", description="Public message visible to all companies (can bluff)")


# ── Observation sub-models ───────────────────────────────────────────────────

class JobView(Observation):
    """Full view of a job (only for the protagonist's own jobs)."""
    id: str = ""
    vram_required: int = 0
    runtime_total: int = 0
    runtime_remaining: int = 0
    priority: int = 1
    assigned_gpu: Optional[str] = None


class CompanyView(Observation):
    """Full view of the protagonist's own company."""
    id: str = ""
    pending_jobs: list[JobView] = Field(default_factory=list)
    running_jobs: list[JobView] = Field(default_factory=list)
    num_completed: int = 0
    starvation_counter: int = 0
    reputation: float = 0.5


class GPUPublicView(Observation):
    """Public info about a GPU (aggregate only, no per-job breakdown)."""
    id: str = ""
    total_vram: int = 0
    free_vram: int = 0
    num_jobs: int = 0


class CompanyPublicView(Observation):
    """Limited view of another company."""
    id: str = ""
    num_pending: int = 0
    num_running: int = 0
    num_completed: int = 0
    starvation_counter: int = 0
    reputation: float = 0.5


class ProposalView(Observation):
    """A proposal received by the protagonist."""
    id: str = ""
    from_company: str = ""
    type: str = ""
    gpu_id: Optional[str] = None
    vram_offered: int = 0
    message: str = ""


class SignalView(Observation):
    """A public signal broadcast by another company."""
    company_id: str = ""
    message: str = ""


class RoundResultsView(Observation):
    """Summary of what happened in the last round."""
    jobs_completed: list[str] = Field(default_factory=list)
    allocations_granted: int = 0
    allocations_denied: int = 0
    coalitions_formed: int = 0


class RewardBreakdownView(Observation):
    """Breakdown of the reward components."""
    priority_progress: float = 0.0
    idle_vram_penalty: float = 0.0
    starvation_penalty: float = 0.0
    coalition_bonus: float = 0.0
    total: float = 0.0


class VRAMObservation(Observation):
    """Partially observable view of the VRAM Exchange world."""
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps in episode")
    my_company: CompanyView = Field(default_factory=CompanyView)
    gpu_states: list[GPUPublicView] = Field(default_factory=list)
    other_companies: list[CompanyPublicView] = Field(default_factory=list)
    incoming_proposals: list[ProposalView] = Field(default_factory=list)
    signals: list[SignalView] = Field(default_factory=list)
    round_results: RoundResultsView = Field(default_factory=RoundResultsView)
    reward_breakdown: RewardBreakdownView = Field(default_factory=RewardBreakdownView)
    text_render: str = Field(default="", description="Human-readable text rendering of the observation")
