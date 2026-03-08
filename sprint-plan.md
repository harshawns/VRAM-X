# Sprint Plan

## Phase 0 (45 minutes): Lock the MVP
**Deliverable:** 1-page spec finalized.
**Before coding, define:** observation fields, actions, reward, episode termination, and one example episode.
**Output:** a short design doc with state, action, reward, and one trace. If you skip this, you will drift.

## Phase 1 (2 hours): Build the World Model
**Implement:** `Job`, `CompanyState`, `GPUState`, `GlobalState`.
**Generator requirements:** seeded episode generator with number of GPUs, VRAM capacities, company job queues, random priorities, random runtimes, and random VRAM requirements.
**Deliverable:** a Python script that prints a valid episode state.
**Done when:** you can generate 3 companies, 10-20 total jobs, 2-4 GPUs, and deterministic runs with a seed.

## Phase 2 (2.5 hours): Build the Environment Loop
**Implement:** `reset()`, `step(actions)`, schedule resolution, resource updates, runtime decrement, completion logic, reward computation.
**Step definition:** one negotiation + allocation cycle followed by one execution tick.
**Deliverable:** you can manually step through an episode in terminal.
**Done when:** you can run reset, then 5-10 steps, and get observations, rewards, and done flag.

## Phase 3 (1.5 hours): Add Baselines
**Implement:** random, greedy selfish, cooperative heuristic.
**Deliverable:** baseline evaluation script reporting average reward over 20 episodes, average idle VRAM, average starvation, and average priority-completion.
**Done when:** you can show random is bad, greedy beats random but is unfair, and cooperative is best or at least meaningfully different in the right way. That alone is a strong demo point.

## Phase 4 (1.5 hours): Make the Environment Feel Real
Add exactly one realism feature: fragmentation, heterogeneous GPU capacity, or message/proposal board.
**Recommended:** heterogeneous GPU capacity or a simple proposal board. Do not add more than one.
**Deliverable:** one extra feature that makes it feel non-toy.

## Phase 5 (2 hours): OpenEnv Wrapper / Packaging
Turn the environment into the format needed for the hackathon.
**Deliverables:** environment package, clean config, example run, and a README containing problem definition, action space, observation space, reward function, and why it matters.
**Done when:** someone can clone the repo and run one episode.

## Phase 6 (2 hours): Minimal Training Script
You do not need massive training.
**Use:** a minimal TRL/OpenEnv-compatible rollout script, tiny training horizon, and a small model or mocked policy if needed.
**Goal:** show reward signal is real and policy can improve a bit, or that imitation/heuristic-assisted improvement is visible.
**Deliverable:** one Colab notebook that loads env, runs episodes, logs rewards, and shows a before/after metric.
If RL gets messy, fall back to supervised imitation of the cooperative baseline and evaluate on env. For the hackathon, the env matters more than perfect RL.

## Phase 7 (1.5 hours): HF Space Deployment
Build a lightweight UI that shows current GPUs, company queues, proposals, selected allocations, and reward metrics.
**UI elements:** company cards, queued jobs, GPU slots with VRAM fill, idle VRAM number, starvation counters, negotiation log.
**Deliverable:** a simple interactive Space.

## Phase 8 (1 hour): Evaluation + Charts
Produce 2-3 simple plots: average reward by baseline, average idle VRAM, average starvation, and maybe priority-completion. These are crucial for storytelling.

## Phase 9 (1 hour): Demo Video
Your 1-minute video should follow this structure:
- **0-10 sec (Problem):** "Shared GPU clusters are scarce. Multiple tenants compete for VRAM, and poor coordination causes idle memory and unfair starvation."
- **10-25 sec (Environment):** "VRAM Exchange is a multi-agent environment where 3-4 companies negotiate over shared GPUs. Each company has private jobs with VRAM, runtime, and priority."
- **25-40 sec (Live environment):** queues, requests, allocation, execution, idle VRAM/starvation metrics.
- **40-50 sec (Baselines):** random, greedy, cooperative/trained.
- **50-60 sec (Close):** "This environment can be reused to train and evaluate multi-agent coordination over scarce compute resources."

## Concrete Task Checklist
**Env core:** config file, state models, seeded generator, observation builder, step function, reward function, done condition.
**Policies:** random, greedy, cooperative heuristic.
**Evaluation:** run 20 episodes; log reward, idle VRAM, starvation, and completion by priority.
**Packaging:** README, screenshots, environment diagram, one sample trace.
**Submission:** public repo, HF Space, Colab notebook, 1-minute YouTube demo.

## Biggest Risks and How to Avoid Them
1. **Too much negotiation logic:** use structured proposals, not free-form text.
2. **Reward is too vague:** keep only 3 terms: priority progress, idle VRAM, starvation.
3. **Training does not really work:** make baselines the main evidence and use minimal training as proof-of-learnability.
4. **Environment feels abstract:** add one realism feature (heterogeneous GPUs or fragmentation).
5. **No clear story:** make the story "poor coordination wastes expensive compute."

## What I Would Personally Build for the MVP
**Version to ship:** 3 companies, 3 GPUs, fixed VRAM capacities, 5 jobs per company, structured request/proposal actions, greedy + cooperative baselines, and `reward = priority progress - idle VRAM - starvation`. That is enough.
**Stretch only if ahead:** GPU heterogeneity, proposal memory, Fleet-style oversight mode.

## Final Recommendation
Do not try to make this a perfect market simulator. Make it a clear reusable environment with scarce resource, private queues, negotiation, and measurable coordination quality.
