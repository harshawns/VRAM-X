"""VRAM Exchange Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import (
    CompanyPublicView,
    CompanyView,
    GPUPublicView,
    GPURequestModel,
    JobView,
    ProposalActionModel,
    ProposalResponseModel,
    ProposalView,
    RewardBreakdownView,
    RoundResultsView,
    SignalView,
    VRAMAction,
    VRAMObservation,
)


class VRAMExchange(EnvClient[VRAMAction, VRAMObservation]):
    """
    Client for the VRAM Exchange Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance controls one protagonist company.

    Example:
        >>> with VRAMExchange(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.text_render)
        ...
        ...     action = VRAMAction(
        ...         requests=[GPURequestModel(job_id="job_0_0", gpu_id="gpu_0", vram_needed=8)],
        ...         signal="I need GPU 0 for a critical job",
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.text_render)
    """

    def _step_payload(self, action: VRAMAction) -> Dict:
        return {
            "requests": [r.model_dump() for r in action.requests],
            "proposals": [p.model_dump() for p in action.proposals],
            "responses": [r.model_dump() for r in action.responses],
            "signal": action.signal,
        }

    def _parse_result(self, payload: Dict) -> StepResult[VRAMObservation]:
        obs_data = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    @staticmethod
    def _parse_observation(obs: Dict, payload: Dict) -> VRAMObservation:
        my_company_data = obs.get("my_company", {})
        my_company = CompanyView(
            id=my_company_data.get("id", ""),
            pending_jobs=[
                JobView(**j) for j in my_company_data.get("pending_jobs", [])
            ],
            running_jobs=[
                JobView(**j) for j in my_company_data.get("running_jobs", [])
            ],
            num_completed=my_company_data.get("num_completed", 0),
            starvation_counter=my_company_data.get("starvation_counter", 0),
            reputation=my_company_data.get("reputation", 0.5),
        )

        gpu_states = [
            GPUPublicView(**g) for g in obs.get("gpu_states", [])
        ]

        other_companies = [
            CompanyPublicView(**c) for c in obs.get("other_companies", [])
        ]

        incoming_proposals = [
            ProposalView(**p) for p in obs.get("incoming_proposals", [])
        ]

        signals = [
            SignalView(**s) for s in obs.get("signals", [])
        ]

        rr = obs.get("round_results", {})
        round_results = RoundResultsView(
            jobs_completed=rr.get("jobs_completed", []),
            allocations_granted=rr.get("allocations_granted", 0),
            allocations_denied=rr.get("allocations_denied", 0),
            coalitions_formed=rr.get("coalitions_formed", 0),
        )

        rb = obs.get("reward_breakdown", {})
        reward_breakdown = RewardBreakdownView(
            priority_progress=rb.get("priority_progress", 0.0),
            idle_vram_penalty=rb.get("idle_vram_penalty", 0.0),
            starvation_penalty=rb.get("starvation_penalty", 0.0),
            coalition_bonus=rb.get("coalition_bonus", 0.0),
            total=rb.get("total", 0.0),
        )

        return VRAMObservation(
            step=obs.get("step", 0),
            max_steps=obs.get("max_steps", 20),
            my_company=my_company,
            gpu_states=gpu_states,
            other_companies=other_companies,
            incoming_proposals=incoming_proposals,
            signals=signals,
            round_results=round_results,
            reward_breakdown=reward_breakdown,
            text_render=obs.get("text_render", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
