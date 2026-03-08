"""
Opponent policies for non-protagonist companies in the VRAM Exchange.

Three built-in policies of increasing sophistication:
  - RandomPolicy:      random GPU requests, no proposals
  - GreedyPolicy:      highest-priority-first allocation, no cooperation
  - CooperativePolicy: shares idle capacity, proposes coalitions, yields GPUs
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from uuid import uuid4

from .game_state import Company, GPU, Proposal
from .negotiation import GPURequest


class BasePolicy(ABC):
    @abstractmethod
    def generate_actions(
        self,
        company: Company,
        gpus: dict[str, GPU],
        other_companies: dict[str, Company],
        step: int,
        rng: random.Random,
    ) -> tuple[list[GPURequest], list[Proposal], dict[str, bool]]:
        """
        Returns (gpu_requests, outgoing_proposals, responses_to_inbox).
        responses_to_inbox maps proposal_id -> accept/reject.
        """
        ...


class RandomPolicy(BasePolicy):
    """Pick random pending jobs, request random GPUs. Ignore proposals."""

    def generate_actions(self, company, gpus, other_companies, step, rng):
        requests: list[GPURequest] = []
        responses: dict[str, bool] = {}

        # Randomly accept/reject all incoming proposals (coin flip)
        for prop in company.inbox:
            responses[prop.id] = rng.random() > 0.5

        if not company.job_queue:
            return requests, [], responses

        num_to_request = rng.randint(1, min(3, len(company.job_queue)))
        jobs_to_request = rng.sample(company.job_queue, num_to_request)

        gpu_list = list(gpus.values())
        for job in jobs_to_request:
            gpu = rng.choice(gpu_list)
            requests.append(GPURequest(
                job_id=job.id,
                gpu_id=gpu.id,
                vram_needed=job.vram_required,
                company_id=company.id,
                priority=job.priority,
            ))

        return requests, [], responses


class GreedyPolicy(BasePolicy):
    """
    Request the best-fit GPU for highest-priority pending jobs.
    Reject all proposals (selfish).
    """

    def generate_actions(self, company, gpus, other_companies, step, rng):
        requests: list[GPURequest] = []
        responses: dict[str, bool] = {p.id: False for p in company.inbox}

        sorted_jobs = sorted(company.job_queue, key=lambda j: -j.priority)

        # Track simulated free vram to avoid requesting the same capacity twice
        simulated_free = {gid: g.free_vram for gid, g in gpus.items()}

        for job in sorted_jobs:
            best_gpu = None
            best_waste = float("inf")

            for gid, gpu in gpus.items():
                avail = simulated_free[gid]
                if avail >= job.vram_required:
                    waste = avail - job.vram_required
                    if waste < best_waste:
                        best_waste = waste
                        best_gpu = gid

            if best_gpu is not None:
                requests.append(GPURequest(
                    job_id=job.id,
                    gpu_id=best_gpu,
                    vram_needed=job.vram_required,
                    company_id=company.id,
                    priority=job.priority,
                ))
                simulated_free[best_gpu] -= job.vram_required

        return requests, [], responses


class CooperativePolicy(BasePolicy):
    """
    Best-fit allocation like Greedy, but also:
    - Accepts beneficial proposals
    - Offers to share idle GPU capacity with starving companies
    - Proposes coalitions with high-reputation partners
    """

    def generate_actions(self, company, gpus, other_companies, step, rng):
        requests: list[GPURequest] = []
        proposals: list[Proposal] = []
        responses: dict[str, bool] = {}

        for prop in company.inbox:
            if prop.type in ("share", "yield"):
                responses[prop.id] = True
            elif prop.type == "coalition":
                sender = other_companies.get(prop.from_company)
                responses[prop.id] = sender is not None and sender.reputation >= 0.4
            else:
                responses[prop.id] = rng.random() > 0.3

        # Greedy-style allocation
        sorted_jobs = sorted(company.job_queue, key=lambda j: -j.priority)
        simulated_free = {gid: g.free_vram for gid, g in gpus.items()}

        for job in sorted_jobs:
            best_gpu = None
            best_waste = float("inf")
            for gid, gpu in gpus.items():
                avail = simulated_free[gid]
                if avail >= job.vram_required:
                    waste = avail - job.vram_required
                    if waste < best_waste:
                        best_waste = waste
                        best_gpu = gid

            if best_gpu is not None:
                requests.append(GPURequest(
                    job_id=job.id,
                    gpu_id=best_gpu,
                    vram_needed=job.vram_required,
                    company_id=company.id,
                    priority=job.priority,
                ))
                simulated_free[best_gpu] -= job.vram_required

        # Offer idle GPU capacity to starving companies
        for oc_id, oc in other_companies.items():
            if oc.starvation_counter < 2:
                continue
            for gid, gpu in gpus.items():
                spare = simulated_free.get(gid, 0)
                if spare >= 8:
                    proposals.append(Proposal(
                        id=str(uuid4())[:8],
                        from_company=company.id,
                        to_company=oc_id,
                        type="share",
                        gpu_id=gid,
                        vram_offered=spare,
                        message=f"Offering {spare}GB on {gid} since you seem stuck",
                    ))
                    break

        # Propose coalition with high-rep partners
        for oc_id, oc in other_companies.items():
            if oc.reputation >= 0.6 and rng.random() > 0.6:
                proposals.append(Proposal(
                    id=str(uuid4())[:8],
                    from_company=company.id,
                    to_company=oc_id,
                    type="coalition",
                    message="Let's coordinate our GPU usage",
                ))

        return requests, proposals, responses


POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "cooperative": CooperativePolicy,
}


def get_policy(name: str) -> BasePolicy:
    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(POLICY_REGISTRY)}")
    return cls()
