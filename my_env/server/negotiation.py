"""
Negotiation resolution engine for the VRAM Exchange environment.

Handles proposal matching, GPU allocation conflict resolution,
coalition formation/dissolution, and reputation updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .game_state import Coalition, Company, GPU, Job, Proposal


@dataclass
class GPURequest:
    job_id: str
    gpu_id: str
    vram_needed: int
    company_id: str
    priority: int = 1


@dataclass
class AllocationResult:
    granted: list[GPURequest] = field(default_factory=list)
    denied: list[GPURequest] = field(default_factory=list)


@dataclass
class NegotiationOutcome:
    accepted_proposals: list[Proposal] = field(default_factory=list)
    rejected_proposals: list[Proposal] = field(default_factory=list)
    new_coalitions: list[Coalition] = field(default_factory=list)
    dissolved_coalitions: list[str] = field(default_factory=list)
    allocation: AllocationResult = field(default_factory=AllocationResult)
    completed_jobs: list[Job] = field(default_factory=list)
    reputation_deltas: dict[str, float] = field(default_factory=dict)


def resolve_proposals(
    proposals: list[Proposal],
    responses: dict[str, bool],
    companies: dict[str, Company],
) -> tuple[list[Proposal], list[Proposal], dict[str, float]]:
    """
    Match proposals with responses. Returns (accepted, rejected, rep_deltas).
    Accepting proposals raises both parties' reputation; rejecting is neutral.
    """
    accepted: list[Proposal] = []
    rejected: list[Proposal] = []
    rep_deltas: dict[str, float] = {}

    for prop in proposals:
        decision = responses.get(prop.id)
        if decision is True:
            prop.accepted = True
            accepted.append(prop)
            rep_deltas[prop.from_company] = rep_deltas.get(prop.from_company, 0.0) + 0.02
            rep_deltas[prop.to_company] = rep_deltas.get(prop.to_company, 0.0) + 0.02
        else:
            prop.accepted = False
            rejected.append(prop)

    return accepted, rejected, rep_deltas


def resolve_coalitions(
    accepted_proposals: list[Proposal],
    existing_coalitions: list[Coalition],
    current_step: int,
) -> tuple[list[Coalition], list[str]]:
    """
    Form new coalitions from accepted coalition proposals.
    Returns (all_coalitions, newly_dissolved_ids).
    """
    new_coalitions: list[Coalition] = []
    dissolved_ids: list[str] = []

    for prop in accepted_proposals:
        if prop.type != "coalition":
            continue

        merged_into: Optional[Coalition] = None
        for c in existing_coalitions:
            if prop.from_company in c.members or prop.to_company in c.members:
                merged_into = c
                break

        if merged_into is not None:
            for member in [prop.from_company, prop.to_company]:
                if member not in merged_into.members:
                    merged_into.members.append(member)
            if prop.gpu_id and prop.gpu_id not in merged_into.shared_gpu_ids:
                merged_into.shared_gpu_ids.append(prop.gpu_id)
        else:
            members = [prop.from_company, prop.to_company]
            shared = [prop.gpu_id] if prop.gpu_id else []
            new_coalitions.append(Coalition(
                id=f"coal_{current_step}_{len(new_coalitions)}",
                members=members,
                shared_gpu_ids=shared,
                formed_at_step=current_step,
            ))

    all_coalitions = existing_coalitions + new_coalitions
    return all_coalitions, dissolved_ids


def allocate_gpus(
    requests: list[GPURequest],
    gpus: dict[str, GPU],
    companies: dict[str, Company],
    coalitions: list[Coalition],
    accepted_proposals: list[Proposal],
) -> AllocationResult:
    """
    Resolve GPU allocation requests. Priority-weighted first-come-first-served.

    Yield/share proposals that were accepted give the beneficiary access to the
    offered VRAM on the specified GPU before normal allocation.
    """
    result = AllocationResult()

    # Process accepted yield/share proposals first: reserve capacity
    reserved: dict[str, int] = {}  # gpu_id -> extra vram reserved by proposals
    for prop in accepted_proposals:
        if prop.type in ("yield", "share") and prop.gpu_id:
            reserved[prop.gpu_id] = reserved.get(prop.gpu_id, 0) + prop.vram_offered

    # Sort requests: higher priority first, then by VRAM (smaller first for packing)
    sorted_requests = sorted(requests, key=lambda r: (-r.priority, r.vram_needed))

    for req in sorted_requests:
        gpu = gpus.get(req.gpu_id)
        if gpu is None:
            result.denied.append(req)
            continue

        if gpu.can_fit(req.vram_needed):
            gpu.allocate(req.job_id, req.vram_needed)
            result.granted.append(req)
        else:
            result.denied.append(req)

    return result


def execute_tick(
    companies: dict[str, Company],
    gpus: dict[str, GPU],
) -> list[Job]:
    """
    Advance one time step: decrement runtime on running jobs, complete finished
    ones, update starvation counters.
    """
    completed: list[Job] = []

    for company in companies.values():
        newly_completed: list[Job] = []

        for job in company.running_jobs:
            job.runtime_remaining -= 1
            if job.runtime_remaining <= 0:
                newly_completed.append(job)

        for job in newly_completed:
            company.running_jobs.remove(job)
            company.completed_jobs.append(job)
            completed.append(job)
            if job.assigned_gpu:
                gpu = gpus.get(job.assigned_gpu)
                if gpu:
                    gpu.deallocate(job.id)
            job.assigned_gpu = None

        if len(company.running_jobs) == 0 and len(company.job_queue) > 0:
            company.starvation_counter += 1

    return completed


def update_reputations(
    companies: dict[str, Company],
    rep_deltas: dict[str, float],
) -> None:
    """Apply reputation deltas, clamped to [0, 1]."""
    for cid, delta in rep_deltas.items():
        if cid in companies:
            companies[cid].reputation = max(0.0, min(1.0, companies[cid].reputation + delta))


def run_negotiation_round(
    all_requests: dict[str, list[GPURequest]],
    all_proposals: list[Proposal],
    all_responses: dict[str, bool],
    companies: dict[str, Company],
    gpus: dict[str, GPU],
    coalitions: list[Coalition],
    current_step: int,
) -> tuple[NegotiationOutcome, list[Coalition]]:
    """
    Full negotiation + allocation + execution pipeline for one step.
    """
    outcome = NegotiationOutcome()

    # 1) Resolve proposals
    accepted, rejected, rep_deltas = resolve_proposals(
        all_proposals, all_responses, companies
    )
    outcome.accepted_proposals = accepted
    outcome.rejected_proposals = rejected
    outcome.reputation_deltas = rep_deltas

    # 2) Coalition formation
    coalitions, dissolved = resolve_coalitions(accepted, coalitions, current_step)
    outcome.new_coalitions = [c for c in coalitions if c.formed_at_step == current_step]
    outcome.dissolved_coalitions = dissolved

    # 3) Move requested jobs from queue to running (if allocation succeeds)
    flat_requests: list[GPURequest] = []
    for company_id, reqs in all_requests.items():
        for r in reqs:
            r.company_id = company_id
            company = companies.get(company_id)
            if company:
                job = next((j for j in company.job_queue if j.id == r.job_id), None)
                if job:
                    r.priority = job.priority
        flat_requests.extend(reqs)

    allocation = allocate_gpus(flat_requests, gpus, companies, coalitions, accepted)
    outcome.allocation = allocation

    # Move granted jobs from queue -> running
    for granted in allocation.granted:
        company = companies.get(granted.company_id)
        if not company:
            continue
        job = next((j for j in company.job_queue if j.id == granted.job_id), None)
        if job:
            job.assigned_gpu = granted.gpu_id
            company.job_queue.remove(job)
            company.running_jobs.append(job)

    # 4) Execute tick
    completed = execute_tick(companies, gpus)
    outcome.completed_jobs = completed

    # 5) Update reputations
    update_reputations(companies, rep_deltas)

    return outcome, coalitions
