# Once each part clicks

## After Gymnasium clicks

You should be able to explain:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

in English without using the word Gymnasium.

Then move to Elder Mist manual play.

## After policy clicks

You should be comfortable saying all of these are policies:

```text
you
if/else rules
Q-table
neural network
LLM + prompt
LLM + RAG
whole agent harness
```

Then move to learning.

## After Q-learning clicks

You should understand:

> The agent chooses differently later because its stored estimates changed from experienced reward.

Then inspect why the Q-policy chose an action by looking at ranked Q-values.

## After state representation clicks

Run the ablation:

```text
full
no_sword
no_locks
minimal
```

Then ask:

> What information disappeared that made previously different situations look identical to the policy?

That is the key state-representation question.

## After POMDP / belief clicks

You should distinguish:

```text
true hidden state
observation
history
belief
```

Then translate that to hidden employee intent / task context / external world facts.

## After neural-weight learning clicks

You should be able to point at:

```text
W1
W2
loss
gradients
weight update
```

and explain what physically changes during training.

Then PPO / OpenAI Five stops being “magic neural RL” and becomes a scaled-up version of familiar pieces.

## After all of that clicks

Return to Fatimah's wording:

> translate sources of data into the best state representations, reconstruct an RL environment to train a harness, configure the best states around it, and make the harness itself a policy.

Read it as:

```text
raw sources
↓
representation candidates
↓
decision system / harness π
↓
actions
↓
interactive environment
↓
reward
↓
learning
↓
held-out evaluation
```

The research questions are then concrete:

1. What sources exist?
2. What information should the representation preserve?
3. What is one action?
4. What is one episode?
5. What is hidden?
6. What is the reward?
7. Which component actually learns?
8. How do we evaluate representations fairly?
9. What is held out?
10. What is the Pareto frontier of performance vs cost?
