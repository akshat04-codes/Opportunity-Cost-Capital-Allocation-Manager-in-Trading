
# Opportunity Cost & Capital Allocation Manager
 
> RL environment focused on **capital allocation decisions** — not price prediction.
 
---
 
## What It Does
 
At each step, the agent receives multiple trading opportunities and must decide **where** and **how much** to invest given limited capital.
 
**Observation:** capital, opportunities (return, risk, confidence), market regime, time remaining  
**Action:** allocation vector — fractions of capital per opportunity, e.g. `[0.5, 0.3, 0.2]`  
**Reward:** profit − risk concentration penalty − idle capital penalty
 
---
 
## Tasks
 
| Task | Opportunities | Steps | Regime Changes |
|---|---|---|---|
| Easy | 4 | 20 | None |
| Medium | 8 | 40 | 10% / step |
| Hard | 15 | 60 | 20% / step |
 
---
 
## Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│                        app/                             │
│          UI / API layer (Streamlit / FastAPI)           │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │       main.py         │
          │     Entry point       │
          └───────────┬───────────┘
                      │
        ┌─────────────▼──────────────┐
        │        config/             │
        │  Hyperparams, env settings │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │        agents/             │
        │   baseline.py              │
        │   Sharpe-weighted agent    │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │          env/              │
        │   Trading Environment      │
        │   reset → step → reward    │
        └──────┬──────────┬──────────┘
               │          │
   ┌───────────▼──┐  ┌────▼────────────┐
   │  inference/  │  │  evaluation/    │
   │  Production  │  │  Benchmarking   │
   │  decisions   │  │  & metrics      │
   └───────────┬──┘  └────┬────────────┘
               │           │
        ┌──────▼───────────▼──────┐
        │        models/          │
        │  Typed dataclasses      │
        │  Opportunity, TaskResult│
        └─────────────────────────┘
```
 
**Episode loop:**
 
```
config/ → env.reset()
  └─► while not done:
        obs     = env.get_opportunities_as_list()
        capital = env.get_available_capital()
        action  = agent.select_action(env, regime)   # agents/
        result  = env.step(action)                   # env/
        reward += result.reward
  └─► evaluation/ → score (0.0 – 1.0)
  └─► inference/  → deploy decisions
```
 
---
 
## Project Structure
 
```
RL/
├── agents/
│   └── baseline.py         # Sharpe-weighted allocation agent
├── app/
│   └── ...                 # UI / API layer (Streamlit / FastAPI / Gradio)
├── config/
│   └── ...                 # Hyperparameters, environment and experiment configs
├── env/
│   └── env.py              # Core environment — state, step, reward
├── evaluation/
│   └── ...                 # Benchmarking — runs agents on tasks, computes metrics
├── inference/
│   └── ...                 # Production inference — decisions only, no training
├── models/
│   └── models.py           # Typed dataclasses (Opportunity, TaskResult, etc.)
├── Dockerfile              # Container setup
├── main.py                 # Entry point
├── openenv.yaml            # OpenEnv specification
└── requirements.txt
```
 
---
 
## Run Locally
 
```bash
pip install -r requirements.txt
python main.py
```
 
```bash
python main.py --seed 42    # set seed
python main.py --quiet      # summaries only
```
 
## Run with Docker
 
```bash
docker build -t rl-trading .
docker run rl-trading
```
 
---
 
## Grader
 
Composite score per task in **[0.0 – 1.0]**:
 
- **55%** Capital growth
- **25%** Reward consistency (Sharpe)
- **20%** Max drawdown penalty
Task weights: Easy `0.20` · Medium `0.35` · Hard `0.45`
