# START HERE — zero to Gymnasium to Elder Mist RL

You do **not** need Sea of Stars installed for the main lab.

You need:
- Windows laptop (your Zenbook is fine)
- Python 3.11 or 3.12
- about 5 minutes for setup
- no GPU

## 1. Check Python

Open **PowerShell** and run:

```powershell
python --version
```

You want something like:

```text
Python 3.11.x
```

or:

```text
Python 3.12.x
```

If `python` is not found, install Python from python.org, then reopen PowerShell.

## 2. Enter this folder

If you downloaded this repo as a ZIP:

```powershell
cd $HOME\Downloads\sea-of-stars-agents-lab-full
```

If the folder has another name, use that name instead.

## 3. Make an isolated Python environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with:

```text
(.venv)
```

If PowerShell blocks activation, run this once for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Install the packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

What those packages are:

```text
numpy       arrays + math
pandas      experiment tables
matplotlib  graphs
gymnasium   standard RL environment API
ipywidgets  clickable notebook buttons
jupyterlab  notebook interface
```

Nothing here requires CUDA.

## 5. Start Jupyter

```powershell
python -m jupyter lab
```

Your browser opens.

## 6. Do the notebooks in THIS order

### First: Gymnasium from zero

```text
notebooks/00_Gym_Basics.ipynb
```

Learn:
- `env.reset()`
- `env.step(action)`
- state
- observation
- action
- transition
- reward
- termination
- policy
- Q-learning

### Second: Elder Mist interactive lab

```text
notebooks/Elder_Mist_RL_Lab.ipynb
```

You manually play the replica first, then:
- run a rule policy
- train Q-learning headlessly
- watch learned behavior
- inspect why an action was chosen
- compare state representations

### Third: belief state / POMDP

```text
notebooks/04_POMDP_Belief_State.ipynb
```

Learn:
- true hidden state
- noisy observation
- history
- belief state
- Bayesian update

### Fourth: weight-level neural learning

```text
notebooks/05_Weight_Level_Learning.ipynb
```

Learn:
- input features
- hidden layer
- logits
- softmax
- loss
- backpropagation
- changing actual neural-network weights

### Fifth: Fatimah / Observable Intuition bridge

```text
notebooks/06_State_Representation_Research.ipynb
```

Turn the Elder Mist experiment into:

```text
raw sources
→ candidate state representations
→ same policy-learning harness
→ held-out task evaluation
→ Pareto frontier
```

That is the direct bridge to screenshot + accessibility tree + history.

## 7. Optional extras after it clicks

`extras/pong/`:
- play Toy Pong
- train PPO
- watch PPO
- remove state variables

`notebooks/02_Atari_Pong.ipynb`:
- raw pixels
- CNN
- learned visual representation

Atari is better on your x64 gaming PC or Colab. It is not needed to understand the core ideas.

---

# The sentence to keep repeating

```text
Environment = the thing that reacts to actions.
Policy      = the thing that chooses actions.
State rep   = the information the policy gets to choose from.
Reward      = the learning signal about outcomes.
```
