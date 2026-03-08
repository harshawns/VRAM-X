"""
VRAM Exchange Environment — the main OpenEnv Environment implementation.

A multi-agent GPU negotiation game where one protagonist company (controlled by
the LLM) competes and cooperates with opponent companies (driven by configurable
policies) over scarce GPU/VRAM resources.

Each step() = one full round:
  1. Parse protagonist action
  2. Generate opponent actions
  3. Resolve negotiations + allocate GPUs
  4. Execute tick (jobs run, complete)
  5. Compute reward
  6. Build partially-observable observation
"""

from __future__ import annotations

import random
from copy import deepcopy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    CompanyPublicView,
    CompanyView,
    GPUPublicView,
    GPURequestModel,
    JobView,
    ProposalView,
    RewardBreakdownView,
    RoundResultsView,
    SignalView,
    VRAMAction,
    VRAMObservation,
)

from .game_state import Coalition, Company, GameConfig, GPU, Proposal, generate_episode
from .negotiation import GPURequest, NegotiationOutcome, run_negotiation_round
from .policies import get_policy


class VRAMEnvironment(Environment):
    """
    Multi-agent VRAM negotiation environment.

    The protagonist (one company) is controlled externally via step().
    All other companies are driven by the configured opponent policy.
    Observations are partially observable from the protagonist's viewpoint.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: GameConfig | None = None):
        self._config = config or GameConfig()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._companies: dict[str, Company] = {}
        self._gpus: dict[str, GPU] = {}
        self._coalitions: list[Coalition] = []
        self._signals: list[tuple[str, str]] = []  # (company_id, message)
        self._last_outcome: NegotiationOutcome | None = None
        self._rng = random.Random(self._config.seed)
        self._opponent_policy = get_policy(self._config.opponent_policy)

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(self) -> VRAMObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(self._config.seed)
        self._coalitions = []
        self._signals = []
        self._last_outcome = None

        companies, gpus = generate_episode(self._config)
        self._companies = {c.id: c for c in companies}
        self._gpus = {g.id: g for g in gpus}

        return self._build_observation(reward_breakdown=RewardBreakdownView())

    def step(self, action: VRAMAction) -> VRAMObservation:  # type: ignore[override]
        self._state.step_count += 1
        step = self._state.step_count
        protagonist = self._companies[self._config.protagonist_id]

        # ── 1. Parse protagonist action ───────────────────────────────────
        proto_requests = [
            GPURequest(
                job_id=r.job_id,
                gpu_id=r.gpu_id,
                vram_needed=r.vram_needed,
                company_id=protagonist.id,
            )
            for r in action.requests
        ]

        proto_proposals = [
            Proposal(
                id=str(uuid4())[:8],
                from_company=protagonist.id,
                to_company=p.to_company,
                type=p.type,
                gpu_id=p.gpu_id,
                vram_offered=p.vram_offered,
                message=p.message,
            )
            for p in action.proposals
        ]

        proto_responses = {r.proposal_id: r.accept for r in action.responses}

        if action.signal:
            self._signals.append((protagonist.id, action.signal))

        # ── 2. Deliver proposals to inboxes ───────────────────────────────
        for prop in proto_proposals:
            target = self._companies.get(prop.to_company)
            if target:
                target.inbox.append(prop)

        # ── 3. Generate opponent actions ──────────────────────────────────
        all_requests: dict[str, list[GPURequest]] = {protagonist.id: proto_requests}
        all_proposals = list(proto_proposals)
        all_responses = dict(proto_responses)

        for cid, company in self._companies.items():
            if cid == self._config.protagonist_id:
                continue

            other = {k: v for k, v in self._companies.items() if k != cid}
            opp_requests, opp_proposals, opp_responses = self._opponent_policy.generate_actions(
                company, self._gpus, other, step, self._rng,
            )

            all_requests[cid] = opp_requests
            all_proposals.extend(opp_proposals)
            all_responses.update(opp_responses)

            for prop in opp_proposals:
                target = self._companies.get(prop.to_company)
                if target:
                    target.inbox.append(prop)

            if hasattr(self._opponent_policy, "signal"):
                pass  # future: opponent signals

        # ── 4. Run negotiation + allocation + execution ───────────────────
        outcome, self._coalitions = run_negotiation_round(
            all_requests=all_requests,
            all_proposals=all_proposals,
            all_responses=all_responses,
            companies=self._companies,
            gpus=self._gpus,
            coalitions=self._coalitions,
            current_step=step,
        )
        self._last_outcome = outcome

        # ── 5. Compute reward ─────────────────────────────────────────────
        reward_breakdown = self._compute_reward(protagonist, outcome)

        # ── 6. Clear inboxes for next round ───────────────────────────────
        for company in self._companies.values():
            company.inbox = []

        # ── 7. Build observation ──────────────────────────────────────────
        obs = self._build_observation(reward_breakdown)
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ── Reward computation ────────────────────────────────────────────────

    def _compute_reward(
        self, protagonist: Company, outcome: NegotiationOutcome
    ) -> RewardBreakdownView:
        w = self._config.reward_weights

        # Priority progress: sum of priorities of jobs completed THIS tick by protagonist
        proto_completed = [j for j in outcome.completed_jobs if j.owner == protagonist.id]
        priority_progress = sum(j.priority for j in proto_completed)

        # Idle VRAM ratio (cluster-wide)
        total_vram = sum(g.total_vram for g in self._gpus.values())
        total_free = sum(g.free_vram for g in self._gpus.values())
        idle_vram_ratio = total_free / max(total_vram, 1)

        # Starvation penalty: sum of starvation counters across ALL companies (fairness)
        starvation_sum = sum(c.starvation_counter for c in self._companies.values())
        starvation_penalty = min(starvation_sum * 0.1, 2.0)

        # Coalition bonus: if protagonist is in any coalition that has active members
        coalition_bonus = 0.0
        for coal in self._coalitions:
            if protagonist.id in coal.members and len(coal.members) >= 2:
                coalition_bonus += 0.5

        total = (
            w.priority_progress * priority_progress
            - w.idle_vram * idle_vram_ratio
            - w.starvation * starvation_penalty
            + w.coalition_bonus * coalition_bonus
        )

        return RewardBreakdownView(
            priority_progress=round(w.priority_progress * priority_progress, 3),
            idle_vram_penalty=round(w.idle_vram * idle_vram_ratio, 3),
            starvation_penalty=round(w.starvation * starvation_penalty, 3),
            coalition_bonus=round(w.coalition_bonus * coalition_bonus, 3),
            total=round(total, 3),
        )

    # ── Observation builder ───────────────────────────────────────────────

    def _build_observation(self, reward_breakdown: RewardBreakdownView) -> VRAMObservation:
        protagonist = self._companies[self._config.protagonist_id]
        step = self._state.step_count

        # Full view of own company
        my_company = CompanyView(
            id=protagonist.id,
            pending_jobs=[
                JobView(
                    id=j.id,
                    vram_required=j.vram_required,
                    runtime_total=j.runtime_total,
                    runtime_remaining=j.runtime_remaining,
                    priority=j.priority,
                    assigned_gpu=j.assigned_gpu,
                )
                for j in protagonist.job_queue
            ],
            running_jobs=[
                JobView(
                    id=j.id,
                    vram_required=j.vram_required,
                    runtime_total=j.runtime_total,
                    runtime_remaining=j.runtime_remaining,
                    priority=j.priority,
                    assigned_gpu=j.assigned_gpu,
                )
                for j in protagonist.running_jobs
            ],
            num_completed=len(protagonist.completed_jobs),
            starvation_counter=protagonist.starvation_counter,
            reputation=round(protagonist.reputation, 2),
        )

        # Public GPU view
        gpu_states = [
            GPUPublicView(
                id=g.id,
                total_vram=g.total_vram,
                free_vram=g.free_vram,
                num_jobs=len(g.allocated_jobs),
            )
            for g in self._gpus.values()
        ]

        # Other companies: public view only
        other_companies = [
            CompanyPublicView(
                id=c.id,
                num_pending=len(c.job_queue),
                num_running=len(c.running_jobs),
                num_completed=len(c.completed_jobs),
                starvation_counter=c.starvation_counter,
                reputation=round(c.reputation, 2),
            )
            for c in self._companies.values()
            if c.id != self._config.protagonist_id
        ]

        # Incoming proposals addressed to protagonist (from last round's opponent actions)
        incoming_proposals = [
            ProposalView(
                id=p.id,
                from_company=p.from_company,
                type=p.type,
                gpu_id=p.gpu_id,
                vram_offered=p.vram_offered,
                message=p.message,
            )
            for p in protagonist.inbox
        ]

        # Public signals
        signals = [
            SignalView(company_id=cid, message=msg)
            for cid, msg in self._signals
            if cid != self._config.protagonist_id
        ]
        # Keep only signals from the latest 2 rounds
        self._signals = self._signals[-(self._config.num_companies * 2):]

        # Round results
        round_results = RoundResultsView()
        if self._last_outcome:
            round_results = RoundResultsView(
                jobs_completed=[j.id for j in self._last_outcome.completed_jobs],
                allocations_granted=len(self._last_outcome.allocation.granted),
                allocations_denied=len(self._last_outcome.allocation.denied),
                coalitions_formed=len(self._last_outcome.new_coalitions),
            )

        # Check done
        all_done = all(c.all_jobs_done for c in self._companies.values())
        done = all_done or step >= self._config.max_steps

        text = self._render_text(my_company, gpu_states, other_companies,
                                 incoming_proposals, signals, round_results,
                                 reward_breakdown, step)

        return VRAMObservation(
            step=step,
            max_steps=self._config.max_steps,
            my_company=my_company,
            gpu_states=gpu_states,
            other_companies=other_companies,
            incoming_proposals=incoming_proposals,
            signals=signals,
            round_results=round_results,
            reward_breakdown=reward_breakdown,
            text_render=text,
            done=done,
            reward=reward_breakdown.total,
        )

    # ── Text rendering ────────────────────────────────────────────────────

    def _render_text(
        self,
        my_company: CompanyView,
        gpu_states: list[GPUPublicView],
        other_companies: list[CompanyPublicView],
        incoming_proposals: list[ProposalView],
        signals: list[SignalView],
        round_results: RoundResultsView,
        reward_breakdown: RewardBreakdownView,
        step: int,
    ) -> str:
        lines: list[str] = []
        lines.append(f"=== VRAM Exchange: Step {step}/{self._config.max_steps} ===")
        lines.append("")

        # Own company
        lines.append(f"YOUR COMPANY: {my_company.id} (reputation: {my_company.reputation})")
        if my_company.pending_jobs:
            pending_strs = [
                f"{j.id} (pri={j.priority}, {j.vram_required}GB, {j.runtime_remaining} ticks)"
                for j in my_company.pending_jobs
            ]
            lines.append(f"  Pending: [{', '.join(pending_strs)}]")
        else:
            lines.append("  Pending: none")

        if my_company.running_jobs:
            running_strs = [
                f"{j.id} on {j.assigned_gpu} ({j.runtime_remaining} ticks left)"
                for j in my_company.running_jobs
            ]
            lines.append(f"  Running: [{', '.join(running_strs)}]")
        else:
            lines.append("  Running: none")

        lines.append(f"  Completed: {my_company.num_completed} jobs")
        if my_company.starvation_counter > 0:
            lines.append(f"  WARNING: Starved for {my_company.starvation_counter} rounds!")
        lines.append("")

        # GPU status
        lines.append("GPU STATUS:")
        for g in gpu_states:
            status = " (IDLE)" if g.free_vram == g.total_vram else ""
            lines.append(f"  {g.id} ({g.total_vram}GB): {g.free_vram}GB free, {g.num_jobs} jobs{status}")
        lines.append("")

        # Other companies
        lines.append("OTHER COMPANIES:")
        for oc in other_companies:
            lines.append(
                f"  {oc.id}: {oc.num_pending} pending, {oc.num_running} running, "
                f"{oc.num_completed} done (rep: {oc.reputation}, starved: {oc.starvation_counter})"
            )
        lines.append("")

        # Incoming proposals
        if incoming_proposals:
            lines.append("INCOMING PROPOSALS:")
            for p in incoming_proposals:
                gpu_info = f" on {p.gpu_id}" if p.gpu_id else ""
                vram_info = f" ({p.vram_offered}GB)" if p.vram_offered else ""
                lines.append(f"  [{p.id}] {p.from_company} proposes {p.type}{gpu_info}{vram_info}: \"{p.message}\"")
            lines.append("")

        # Signals
        if signals:
            lines.append("SIGNALS:")
            for s in signals:
                lines.append(f"  {s.company_id}: \"{s.message}\"")
            lines.append("")

        # Round results
        if step > 0:
            lines.append("LAST ROUND:")
            lines.append(f"  Allocations: {round_results.allocations_granted} granted, {round_results.allocations_denied} denied")
            if round_results.jobs_completed:
                lines.append(f"  Jobs completed: {', '.join(round_results.jobs_completed)}")
            if round_results.coalitions_formed:
                lines.append(f"  New coalitions: {round_results.coalitions_formed}")
            lines.append("")

        # Reward
        lines.append("REWARD:")
        lines.append(f"  Priority progress: +{reward_breakdown.priority_progress}")
        lines.append(f"  Idle VRAM penalty: -{reward_breakdown.idle_vram_penalty}")
        lines.append(f"  Starvation penalty: -{reward_breakdown.starvation_penalty}")
        lines.append(f"  Coalition bonus: +{reward_breakdown.coalition_bonus}")
        lines.append(f"  TOTAL: {reward_breakdown.total}")

        return "\n".join(lines)
