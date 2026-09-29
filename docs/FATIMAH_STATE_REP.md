# Mapping this lab to Fatimah's state-representation research

The Elder Mist lab is intentionally a miniature version of the research problem.

## Elder Mist

```text
true simulator state
        ↓
candidate representation
        ↓
harness / policy
        ↓
action
        ↓
environment transition
        ↓
reward
```

## Computer-use / Observable Intuition

```text
true computer + user + task world
        ↓
raw sources
    screenshot
    accessibility tree
    mouse / keyboard
    application data
    history
        ↓
candidate state representation
        ↓
harness / policy
        ↓
computer action
        ↓
world changes
        ↓
task reward / success
```

## Research question 1: state representation

Not:

> capture as much telemetry as possible.

Instead:

> Which transformation of the available sources preserves the information needed to choose good actions?

Candidate representations:

```text
A = screenshot only
B = accessibility tree only
C = screenshot + tree
D = screenshot + tree + last N actions
E = structured inferred task state
F = learned latent representation
```

## Research question 2: environment

You need a loop that can repeatedly do:

```text
observe
→ act
→ get consequence
→ observe again
→ score outcome
```

τ² already gives you a controlled version of this for customer-service tasks.

This Elder Mist simulator gives you another controlled version.

## Research question 3: learning the harness/policy

"Train the harness" is underspecified until you answer:

- what component has parameters?
- what receives gradients / updates?
- what is the reward?
- what is one episode?
- what counts as an action?
- what is held fixed?

Possible things to optimize:
- model weights
- LoRA weights
- state encoder
- retrieval policy
- memory selection
- planner
- tool-selection policy
- prompt/state template
- combinations of these

## How to measure "best state"

Do not ask an LLM which representation sounds nicest.

Hold the tasks and policy-learning method fixed, then measure:

```text
task success
steps
latency
tokens
tool errors
policy violations
generalization to held-out tasks
```

A representation is useful if it improves downstream behavior.

## Pareto frontier

"Best" may be multi-objective.

Example:

```text
Representation A: 88% success, 10k tokens
Representation B: 87% success, 2k tokens
```

B may be preferable even with slightly lower success.

The Pareto frontier contains representations for which no other candidate is better on all relevant objectives.

## The experiment to propose

1. Define task split.
2. Pre-register metrics.
3. Generate several state representations from the same raw sources.
4. Keep the harness/model fixed.
5. Evaluate on development tasks.
6. Iterate.
7. Freeze the design.
8. Evaluate once on held-out tasks.
9. Compare success/cost/error metrics.
10. Only then ask whether learned state encoders beat hand-designed ones.
