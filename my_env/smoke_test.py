"""
Smoke test: run the VRAM Exchange environment directly (no server).

Resets the environment and steps through 10 rounds with a mix of
requests, proposals, and signals to verify the full loop works.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models import GPURequestModel, ProposalActionModel, ProposalResponseModel, VRAMAction, VRAMObservation
from server.game_state import GameConfig
from server.vram_environment import VRAMEnvironment


def run():
    config = GameConfig(
        num_companies=3,
        num_gpus=3,
        jobs_per_company=5,
        max_steps=15,
        gpu_capacities=[24, 48, 80],
        protagonist_id="company_0",
        opponent_policy="greedy",
        seed=42,
    )

    env = VRAMEnvironment(config)
    obs = env.reset()

    print("=" * 70)
    print("INITIAL STATE")
    print("=" * 70)
    print(obs.text_render)
    print()

    for step_num in range(1, config.max_steps + 1):
        # Build a simple action: request GPUs for pending jobs
        requests = []
        my = obs.my_company
        gpu_states = obs.gpu_states

        for job in my.pending_jobs[:2]:
            for gpu in gpu_states:
                if gpu.free_vram >= job.vram_required:
                    requests.append(GPURequestModel(
                        job_id=job.id,
                        gpu_id=gpu.id,
                        vram_needed=job.vram_required,
                    ))
                    break

        # Respond to any incoming proposals (accept all)
        responses = [
            ProposalResponseModel(proposal_id=p.id, accept=True)
            for p in obs.incoming_proposals
        ]

        # Propose coalition with company_1 on step 3
        proposals = []
        if step_num == 3:
            proposals.append(ProposalActionModel(
                to_company="company_1",
                type="coalition",
                message="Let's team up on GPU allocation",
            ))

        signal = ""
        if step_num == 1:
            signal = "Looking to cooperate on GPU sharing"

        action = VRAMAction(
            requests=requests,
            proposals=proposals,
            responses=responses,
            signal=signal,
        )

        obs = env.step(action)

        print("=" * 70)
        print(f"STEP {step_num}")
        print("=" * 70)
        print(obs.text_render)
        print()

        if obs.done:
            print(f"Episode ended at step {step_num}.")
            break

    # Final summary
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  Steps taken: {env.state.step_count}")
    print(f"  Final reward: {obs.reward}")
    print(f"  My completed jobs: {obs.my_company.num_completed}")
    print(f"  My pending jobs: {len(obs.my_company.pending_jobs)}")
    print(f"  My running jobs: {len(obs.my_company.running_jobs)}")
    print(f"  Done: {obs.done}")


if __name__ == "__main__":
    run()
