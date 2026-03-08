"""
Core data models and episode generator for the VRAM Exchange environment.

Defines Job, GPU, Company, Proposal, Coalition, GameConfig, and the seeded
episode generator that produces contention-rich starting scenarios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class Job:
    id: str
    vram_required: int
    runtime_total: int
    runtime_remaining: int
    priority: int  # 1 (low) .. 5 (critical)
    owner: str
    assigned_gpu: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.assigned_gpu is not None and self.runtime_remaining > 0

    @property
    def is_completed(self) -> bool:
        return self.runtime_remaining <= 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vram_required": self.vram_required,
            "runtime_total": self.runtime_total,
            "runtime_remaining": self.runtime_remaining,
            "priority": self.priority,
            "owner": self.owner,
            "assigned_gpu": self.assigned_gpu,
        }


@dataclass
class GPU:
    id: str
    total_vram: int
    allocated_jobs: list[str] = field(default_factory=list)

    _job_vram_map: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def used_vram(self) -> int:
        return sum(self._job_vram_map.values())

    @property
    def free_vram(self) -> int:
        return self.total_vram - self.used_vram

    def can_fit(self, vram: int) -> bool:
        return self.free_vram >= vram

    def allocate(self, job_id: str, vram: int) -> None:
        self.allocated_jobs.append(job_id)
        self._job_vram_map[job_id] = vram

    def deallocate(self, job_id: str) -> None:
        if job_id in self._job_vram_map:
            del self._job_vram_map[job_id]
        if job_id in self.allocated_jobs:
            self.allocated_jobs.remove(job_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "total_vram": self.total_vram,
            "free_vram": self.free_vram,
            "used_vram": self.used_vram,
            "num_jobs": len(self.allocated_jobs),
        }

    def to_full_dict(self) -> dict:
        d = self.to_dict()
        d["allocated_jobs"] = list(self.allocated_jobs)
        d["job_vram_map"] = dict(self._job_vram_map)
        return d


@dataclass
class Proposal:
    id: str
    from_company: str
    to_company: str
    type: str  # "share", "trade", "coalition", "yield"
    gpu_id: Optional[str] = None
    vram_offered: int = 0
    job_id: Optional[str] = None
    message: str = ""
    accepted: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_company": self.from_company,
            "to_company": self.to_company,
            "type": self.type,
            "gpu_id": self.gpu_id,
            "vram_offered": self.vram_offered,
            "job_id": self.job_id,
            "message": self.message,
            "accepted": self.accepted,
        }


@dataclass
class Coalition:
    id: str
    members: list[str]
    shared_gpu_ids: list[str] = field(default_factory=list)
    formed_at_step: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "members": list(self.members),
            "shared_gpu_ids": list(self.shared_gpu_ids),
            "formed_at_step": self.formed_at_step,
        }


@dataclass
class Company:
    id: str
    job_queue: list[Job] = field(default_factory=list)
    running_jobs: list[Job] = field(default_factory=list)
    completed_jobs: list[Job] = field(default_factory=list)
    starvation_counter: int = 0
    reputation: float = 0.5
    inbox: list[Proposal] = field(default_factory=list)

    @property
    def all_jobs_done(self) -> bool:
        return len(self.job_queue) == 0 and len(self.running_jobs) == 0

    @property
    def total_pending_vram(self) -> int:
        return sum(j.vram_required for j in self.job_queue)

    @property
    def total_priority_weight(self) -> int:
        return sum(j.priority for j in self.job_queue)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_queue": [j.to_dict() for j in self.job_queue],
            "running_jobs": [j.to_dict() for j in self.running_jobs],
            "completed_jobs": [j.to_dict() for j in self.completed_jobs],
            "starvation_counter": self.starvation_counter,
            "reputation": self.reputation,
            "num_pending": len(self.job_queue),
            "num_running": len(self.running_jobs),
            "num_completed": len(self.completed_jobs),
        }

    def to_public_dict(self) -> dict:
        """Limited view visible to other companies."""
        return {
            "id": self.id,
            "num_pending": len(self.job_queue),
            "num_running": len(self.running_jobs),
            "num_completed": len(self.completed_jobs),
            "starvation_counter": self.starvation_counter,
            "reputation": round(self.reputation, 2),
        }


@dataclass
class RewardWeights:
    priority_progress: float = 1.0
    idle_vram: float = 0.3
    starvation: float = 0.5
    coalition_bonus: float = 0.2


@dataclass
class GameConfig:
    num_companies: int = 3
    num_gpus: int = 3
    jobs_per_company: int = 5
    max_steps: int = 20
    gpu_capacities: list[int] = field(default_factory=lambda: [24, 48, 80])
    protagonist_id: str = "company_0"
    opponent_policy: str = "greedy"
    seed: Optional[int] = None
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    vram_choices: list[int] = field(default_factory=lambda: [4, 8, 16, 24, 48])
    runtime_range: tuple[int, int] = (1, 5)
    priority_range: tuple[int, int] = (1, 5)


def generate_episode(config: GameConfig) -> tuple[list[Company], list[GPU]]:
    """Generate a seeded episode with guaranteed resource contention."""
    rng = random.Random(config.seed)

    gpus = []
    for i in range(config.num_gpus):
        cap = config.gpu_capacities[i % len(config.gpu_capacities)]
        gpus.append(GPU(id=f"gpu_{i}", total_vram=cap))

    total_cluster_vram = sum(g.total_vram for g in gpus)

    companies = []
    for c_idx in range(config.num_companies):
        company_id = f"company_{c_idx}"
        jobs: list[Job] = []

        for j_idx in range(config.jobs_per_company):
            feasible_vram = [v for v in config.vram_choices if v <= max(config.gpu_capacities)]
            vram = rng.choice(feasible_vram)
            rt = rng.randint(*config.runtime_range)
            pri = rng.randint(*config.priority_range)
            jobs.append(Job(
                id=f"job_{company_id}_{j_idx}",
                vram_required=vram,
                runtime_total=rt,
                runtime_remaining=rt,
                priority=pri,
                owner=company_id,
            ))

        companies.append(Company(id=company_id, job_queue=jobs))

    # Guarantee contention: if total demand is not > total supply, inflate a few jobs
    total_demand = sum(c.total_pending_vram for c in companies)
    while total_demand <= total_cluster_vram:
        target_company = rng.choice(companies)
        target_job = rng.choice(target_company.job_queue)
        bump = rng.choice([4, 8, 16])
        new_vram = min(target_job.vram_required + bump, max(config.gpu_capacities))
        target_job.vram_required = new_vram
        total_demand = sum(c.total_pending_vram for c in companies)

    return companies, gpus
