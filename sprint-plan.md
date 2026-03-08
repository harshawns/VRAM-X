# Sprint Plan

## Phase 0 (45 minutes): Lock the MVP

### Deliverable
- 1-page spec finalized

### Write This Down Before Coding
- Observation fields
- Actions
- Reward
- Episode termination
- One example episode

### Output
A short design doc with:
- State
- Action
- Reward
- One trace

If you skip this, you will drift.

## Phase 1 (2 hours): Build the World Model

### Implement
- `Job`
- `CompanyState`
- `GPUState`
- `GlobalState`

Implement a seeded episode generator with:
- Number of GPUs
- VRAM capacities
- Company job queues
- Random priorities
- Random runtimes
- Random VRAM requirements

### Deliverable
- A Python script that prints a valid episode state

### Done When
You can generate:
- 3 companies
- 10-20 jobs total
- 2-4 GPUs
- Deterministic runs with a seed

## Phase 2 (2.5 hours): Build the Environment Loop

### Implement
- `reset()`
- `step(actions)`
- Schedule resolution
- Resource updates
- Runtime decrement
- Completion logic
- Reward computation

### Keep It Simple
A step can mean:
- One negotiation + allocation cycle
- Followed by one execution tick

### Deliverable
- You can manually step through an episode in terminal

### Done When
You can run:
- Reset
- 5-10 steps
- Get observations, rewards, done flag

## Phase 3 (1.5 hours): Add Baselines

### Implement
- Random
- Greedy selfish
- Cooperative heuristic

### Deliverable
A baseline evaluation script with:
- Average reward over 20 episodes
- Average idle VRAM
- Average starvation
- Average priority-completion

### Done When
You can show:
- Random is bad
- Greedy is better than random but unfair
- Cooperative is best, or at least different in the right way

That alone is a strong demo point.

## Phase 4 (1.5 hours): Make the Environment Feel Real

Add exactly one realism feature:
- Fragmentation
- Heterogeneous GPU capacity
- Message/proposal board

Recommended:
- Heterogeneous GPU capacity, or
- Simple proposal board

Do not add more than one.

### Deliverable
- One extra feature that makes it feel non-toy

## Phase 5 (2 hours): OpenEnv Wrapper / Packaging

Turn the environment into the format needed for the hackathon.

### Deliverables
- Environment package
- Clean config
- Example run
- README with:
  - Problem definition
  - Action space
  - Observation space
  - Reward function
  - Why it matters

### Done When
- Someone can clone the repo and run one episode

## Phase 6 (2 hours): Minimal Training Script

You do not need massive training.

Use:
- A minimal TRL / OpenEnv-compatible rollout script
- Tiny training horizon
- Small model or mocked policy if needed

The goal is to show:
- Reward signal is real
- Policy can improve a bit
- Or imitation / heuristic-assisted improvement is visible

### Deliverable
One Colab notebook that:
- Loads env
- Runs episodes
- Logs rewards
- Shows a before/after metric

If RL gets messy, fall back to:
- Supervised imitation of cooperative baseline
- Then evaluate on env

For the hackathon, the env matters more than perfect RL.

## Phase 7 (1.5 hours): HF Space Deployment

Build a lightweight UI to:
- Show current GPUs
- Show company queues
- Show proposals
- Show selected allocations
- Show reward metrics

The UI should help judges understand the world.

### UI Elements
- Company cards
- Queued jobs
- GPU slots with VRAM fill
- Idle VRAM number
- Starvation counters
- Negotiation log

### Deliverable
- A simple interactive Space

## Phase 8 (1 hour): Evaluation + Charts

Produce 2-3 simple plots:
- Average reward by baseline
- Average idle VRAM
- Average starvation
- Maybe priority-completion

These are crucial for storytelling.

## Phase 9 (1 hour): Demo Video

Your 1-minute video should be:

### 0-10 sec
Problem:
"Shared GPU clusters are scarce. Multiple tenants compete for VRAM, and poor coordination causes idle memory and unfair starvation."

### 10-25 sec
Environment:
"VRAM Exchange is a multi-agent environment where 3-4 companies negotiate over shared GPUs. Each company has private jobs with VRAM, runtime, and priority."

### 25-40 sec
Show live environment:
- Queues
- Requests
- Allocation
- Execution
- Idle VRAM / starvation metrics

### 40-50 sec
Show baseline comparison:
- Random
- Greedy
- Cooperative / trained

### 50-60 sec
Close:
"This environment can be reused to train and evaluate multi-agent coordination over scarce compute resources."

## Concrete Task Checklist

### Env Core
- Config file
- State models
- Seeded generator
- Observation builder
- Step function
- Reward function
- Done condition

### Policies
- Random
- Greedy
- Cooperative heuristic

### Evaluation
- Run 20 episodes
- Log reward
- Log idle VRAM
- Log starvation
- Log completion by priority

### Packaging
- README
- Screenshots
- Environment diagram
- One sample trace

### Submission
- Public repo
- HF Space
- Colab notebook
- 1-minute YouTube demo

## Biggest Risks and How to Avoid Them

### Risk 1: Too Much Negotiation Logic
Fix:
- Use structured proposals, not free-form text

### Risk 2: Reward Is Too Vague
Fix:
- Keep only 3 terms:
  - Priority progress
  - Idle VRAM
  - Starvation

### Risk 3: Training Does Not Really Work
Fix:
- Make baselines your main evidence and use minimal training as proof-of-learnability

### Risk 4: Environment Feels Abstract
Fix:
- Add one realism feature:
  - Heterogeneous GPUs
  - Fragmentation

### Risk 5: No Clear Story
Fix:
- Make the story:
  - "Poor coordination wastes expensive compute"

## What I Would Personally Build for the MVP

### Version You Should Actually Ship
- 3 companies
- 3 GPUs
- Fixed VRAM capacities
- 5 jobs per company
- Structured request/proposal actions
- Greedy + cooperative baselines
- `reward = priority progress - idle VRAM - starvation`

That is enough.

### Stretch Only If Ahead
- GPU heterogeneity
- Proposal memory
- Fleet-style oversight mode

## Final Recommendation

Do not try to make this into a perfect market simulator.

Make it a clear reusable environment with:
- Scarce resource
- Private queues
- Negotiation
- Measurable coordination quality
