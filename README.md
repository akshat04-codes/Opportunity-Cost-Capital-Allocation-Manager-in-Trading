# 📊 Opportunity Cost & Capital Allocation Manager in Trading

---

## 🚀 Project Motivation

In real-world trading, success is not only about predicting price movements—it is fundamentally about **decision-making under constraints**.

A trader often faces multiple opportunities at the same time but has:

* limited capital
* limited risk tolerance
* uncertainty in outcomes

This creates the problem of **opportunity cost**:

> Choosing one trade means potentially missing a better one.

Poor decisions can lead to:

* inefficient capital usage
* excessive risk concentration
* missed high-value opportunities

This project simulates this real-world challenge by creating an environment where an AI agent must learn to **allocate capital optimally across competing trading opportunities**.

---

## 🧠 Environment Description

This project implements a **custom OpenEnv-compatible environment** where:

* The agent is given multiple trading opportunities at each step
* Each opportunity has:

  * expected return
  * risk level
  * confidence score

The agent must:

* decide **which opportunities to invest in**
* decide **how much capital to allocate to each**

The environment then:

* simulates outcomes
* updates capital
* returns a reward based on decision quality

👉 The goal is not just profit, but **efficient, disciplined, and balanced allocation of capital over time**.

---

## 🔍 Observation Space

At each step, the agent receives:

* `available_capital` (float)
* `opportunities` (list of objects):

  * `expected_return` (0–1)
  * `risk` (0–1)
  * `confidence` (0–1)
* `open_positions` (list)

This represents the **current decision context**.

---

## 🎯 Action Space

The agent outputs:

* `allocations`: list of floats (0–1)

Constraints:

* Each value represents % of capital allocated to a trade
* Total allocation must be ≤ 1 (i.e., ≤ 100% capital)

Example:

```python
[0.5, 0.3, 0.0]
```

Means:

* 50% capital → Trade A
* 30% capital → Trade B
* skip Trade C

---

## 🏆 Reward Design

The reward function is **continuous (0.0 → 1.0)** and designed to reflect real-world trading quality.

### ✅ Positive Components

* **Profit generation**
* **Efficient capital utilization**
* **Balanced allocation across opportunities**

### ❌ Penalties

* **Risk concentration** (too much capital in high-risk trades)
* **Missed opportunities** (ignoring high-value trades)
* **Poor allocation decisions**

### 📌 Conceptual Formula

```
Reward = Profit Score
       - Risk Penalty
       - Inefficiency Penalty
```

👉 This ensures:

* partial progress is rewarded
* bad behavior is penalized
* long-term consistency is encouraged

---

## 🎮 Tasks

The environment includes **three difficulty levels**:

---

### 🟢 Easy Task

* 2–3 opportunities
* clear optimal choice

**Goal:**
Learn basic selection and avoid poor decisions

---

### 🟡 Medium Task

* 4–5 opportunities
* mixed risk-return profiles

**Goal:**
Balance capital allocation across multiple trades

---

### 🔴 Hard Task

* dynamic opportunities
* changing conditions each step

**Goal:**
Optimize long-term capital growth while managing risk

---

## ⚙️ Setup Instructions

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ 2. Run the Environment (Baseline)

```bash
python baseline.py
```

This will:

* run the agent on all tasks
* output performance metrics

---

### 🐳 Optional: Run with Docker

```bash
docker build -t trading-env .
docker run trading-env
```

---

## 🤖 Baseline Results

A simple baseline agent is implemented with the strategy:

> Allocate more capital to opportunities with high (expected_return / risk)

### 📊 Output Includes:

* final capital
* total reward
* task-wise performance

Example:

```
Easy Task Score:    0.85
Medium Task Score:  0.72
Hard Task Score:    0.60
```

👉 These results provide a **reference benchmark** for evaluating more advanced agents.

---

## 📦 Project Structure

```
project/
│
├── models.py
├── env.py
├── tasks.py
├── grader.py
├── baseline.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🎯 Key Highlights

* Real-world trading simulation (not a toy problem)
* Focus on **decision-making, not prediction**
* Continuous reward system
* Multi-level task evaluation
* Fully modular and extensible design

---

## 📌 Future Improvements

* Add time-series market dynamics
* Include transaction costs
* Introduce portfolio constraints
* Train RL agents on the environment

---

## 🧠 Final Thought

This project captures a fundamental truth in trading:

> “The best decision is not always about profit—it’s about choosing the right opportunity at the right time with the right amount of risk.”

---
