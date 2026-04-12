# Simulating OBELIX: A Behaviour-based Robot

![Teaser image](./OBELIX.png)
**Picture:** *The figure shows the OBELIX robot examining a box, taken from the paper ["Automatic Programming of Behaviour-based Robots using Reinforcement Learning"](https://cdn.aaai.org/AAAI/1991/AAAI91-120.pdf)*


This repo consists of the code for simulating the OBELIX robot, as described in the paper ["Automatic Programming of Behaviour-based Robots using Reinforcement Learning"](https://cdn.aaai.org/AAAI/1991/AAAI91-120.pdf) by Sridhar Mahadevan and Jonathan Connell. The code is written in Python 3.7 and uses the [OpenCV](https://docs.opencv.org/4.x/) library for the GUI.

Some of this codebase is adapted from: https://github.com/iabhinavjoshi/OBELIX

*This repo is used for practicing RL algorithms covered during the NPTEL's course [Reinforcement Learning](https://onlinecourses.nptel.ac.in/noc19_cs55/preview) 2023.*


## 1. Problem Overview

The OBELIX task is a **robotic box-pushing problem** in a 2D arena. The agent must:

1. **Find** the grey box using local sensor observations  
2. **Attach** to it when it comes within range  
3. **Push** it toward the arena boundary  
4. **Unwedge** or recover when the box gets blocked near walls or obstacles  

The robot has only a limited 18-bit binary observation vector and a small discrete action set, so it must learn the task through trial-and-error rather than direct access to positions or maps. This makes the environment a **Partially Observable Markov Decision Process (POMDP)**. :contentReference[oaicite:3]{index=3}

---

## 2. Environment Description

The robot operates in a **500 × 500 pixel arena** with **5 discrete actions**:

- `L45` — turn left 45°
- `L22` — turn left 22.5°
- `FW` — move forward
- `R22` — turn right 22.5°
- `R45` — turn right 45°  

Each forward action moves the robot by 5 pixels. Episodes terminate after a maximum of **2000 steps** or upon successful completion. :contentReference[oaicite:4]{index=4}

The observation space is an **18-dimensional binary vector**:

| Index | Sensor | Description |
|---|---|---|
| 0–3 | Left sonar (×4) | 2 positions × (far, near) range on the left side |
| 4–11 | Forward sonar (×8) | 4 positions × (far, near) range facing forward |
| 12–15 | Right sonar (×4) | 2 positions × (far, near) range on the right side |
| 16 | IR sensor | Forward-facing infrared sensor; detects box at close range |
| 17 | Stuck flag | Set to 1 if the robot is stuck against a wall or obstacle |

The sonar sensors have two ranges: **far** (scaled by 30) and **near** (scaled by 18), each with a **20° field of view**. The IR sensor has a shorter range (scaled by 4) and is critical for attachment. :contentReference[oaicite:5]{index=5}

---

## 3. Difficulty Levels

The environment includes three difficulty levels that progressively increase task complexity:

| Difficulty | Box Behaviour | Description |
|---|---|---|
| 0 (Static) | Stationary | The box remains fixed at its spawn location. The agent only needs to locate and push it to the boundary. |
| 2 (Blinking) | Periodic visibility toggling | The box alternates between visible and invisible periods. The agent must handle intermittent sensor signals and infer the box’s position over time. |
| 3 (Moving + Blinking) | Random motion + blinking | The box moves along a random trajectory while also blinking. The agent must track and intercept a moving, intermittently visible target. |

In addition, the environment may contain an **optional wall obstacle** mode, where wall segments are placed in the arena and can block movement or sensor lines. This increases the challenge further by forcing the agent to recover from getting stuck and to navigate around barriers. :contentReference[oaicite:6]{index=6}

---

## 4. Why This Problem is Challenging

OBELIX is challenging because it combines:

- **Sparse and low-dimensional observations**
- **No direct access to positions or velocities**
- **Partial observability**
- **Blinking and moving targets**
- **Wall occlusions and stuck states**
- **Long-horizon reward dependency**

The agent only sees whether something is inside a sensor cone, not the full state of the world. Because of this, the task is naturally modeled as a **POMDP**, and the agent must use sequential sensor patterns to infer hidden information. :contentReference[oaicite:7]{index=7}

The high-level objective is to **maximize mean cumulative reward** across all difficulty levels while maintaining robustness to wall obstacles and generalization across random seeds. :contentReference[oaicite:8]{index=8}

---

## 5. Reward Structure

The environment provides sparse rewards that reward discovery, attachment, and successful goal completion:

- Sonar activations: small positive rewards
- IR activation: higher reward
- Stuck against wall: large negative reward
- Successful delivery to boundary: large terminal bonus

The report notes that the environment has a strong sparse-reward nature, with the success bonus dominating the return and the per-step signals being relatively weak. This makes learning difficult without reward shaping or improved exploration. :contentReference[oaicite:9]{index=9}

---

## 6. Methods Explored

The project explored multiple reinforcement learning approaches before settling on the best-performing one.

### 6.1 Vanilla DQN
A baseline Deep Q-Network with a simple MLP on the raw 18-bit observations was used first. This baseline struggled because the observation space has no explicit positional information and random exploration often led to wandering or spinning without locating the box. :contentReference[oaicite:10]{index=10}

### 6.2 Double DQN
Double DQN improved stability by reducing Q-value overestimation, but the main issues of partial observability and weak exploration still remained. :contentReference[oaicite:11]{index=11}

### 6.3 Final Best Model: D3QN
The best model was a **Dueling Double Deep Q-Network (D3QN)**, which decomposes Q-values into a **state-value stream** and an **advantage stream**. This helped the agent learn state quality more effectively in large regions of the state space where many actions have similar values. :contentReference[oaicite:12]{index=12}

---

## 7. Key Improvements in the Final System

The final system added several important improvements:

### Prioritized Experience Replay (PER)
Transitions with larger TD error were sampled more often, which was useful because most transitions in OBELIX are uninformative while discovery and attachment events are rare. :contentReference[oaicite:13]{index=13}

### n-step Returns
Using **3-step returns** helped propagate sparse rewards more quickly through the replay targets. :contentReference[oaicite:14]{index=14}

### Observation Stacking
The last **6 observations** were stacked into a **108-dimensional input** to give the model short-term temporal context and reduce partial observability. :contentReference[oaicite:15]{index=15}

### Potential-Based Reward Shaping (PBRS)
A three-component shaping scheme was used:

- bot-to-box distance
- heading alignment toward the box
- box-to-boundary distance after attachment  

This was identified as the most impactful improvement because it transformed the sparse reward problem into a denser learning signal while preserving the optimal policy. :contentReference[oaicite:16]{index=16}

### Forward-Biased Exploration
Instead of uniform random exploration, the policy biased random actions toward forward movement, which improved traversal efficiency. A stuck-aware policy also avoided wasting forward actions when the robot was against a wall. :contentReference[oaicite:17]{index=17}

### Curriculum Learning
Training was organized into four phases:

- warm-up without walls
- wall introduction
- blinking introduction
- full difficulty training  

This staged training schedule was essential because training directly on the hardest setting caused early collapse. :contentReference[oaicite:18]{index=18}

---

## 8. Network Architecture

The final D3QN model used the following structure:

- Input: 108-dim stacked observation
- Backbone: Linear(108 → 256) → ReLU → Linear(256 → 128) → ReLU
- Value stream: 128 → 64 → 1
- Advantage stream: 128 → 64 → 5
- Dueling aggregation:

\[
Q(s,a) = V(s) + \left(A(s,a) - \text{mean}(A)\right)
\]

This network had approximately **47,000 parameters** and was small enough to train efficiently while still capturing useful temporal structure from the stacked observations. :contentReference[oaicite:19]{index=19}

---

## 9. Hyperparameters

The final training configuration used:

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Discount factor | 0.99 |
| Batch size | 64 |
| Replay buffer | 50,000 |
| PER alpha | 0.6 |
| PER beta | 0.4 → 1.0 |
| n-step | 3 |
| Target update | Every 2000 steps |
| Observation stack | 6 frames |
| Shaping scale | 10.0 |

These settings were selected to balance stability, sample efficiency, and reward propagation in the long-horizon task. :contentReference[oaicite:20]{index=20}

---

## 10. Results

The best-performing model achieved the following mean rewards:

| Phase | Best Mean Reward |
|---|---:|
| Phase 1 (Level 1) | -1980.70 |
| Phase 2 (Level 2) | -1977.40 |
| Phase 3 (Level 3) | -1420.32 |
| Test Phase | -1796.41 |

The report concludes that the final D3QN with PER, stacking, PBRS, and curriculum learning performed best overall, especially on the more difficult dynamic settings. :contentReference[oaicite:21]{index=21}

A training dashboard and learning curves were also generated to monitor episode-wise reward progression, loss, success rate, exploration behavior, and phase-wise performance. :contentReference[oaicite:22]{index=22}

---

## 11. Error Analysis and Discussion

Several enhancements were tested but later removed or reverted:

- **Action masking near walls** was reverted because hard masking caused instability.
- **Exploration bonuses** such as RND and count-based rewards were removed because they conflicted with PBRS.
- **PPO** was tested as an alternative, but it did not work as well under the limited episode budget and discrete-action setting.

The report emphasizes that not all algorithmic enhancements compose well; the best results came from a carefully balanced combination of D3QN, PER, n-step returns, stacking, PBRS, and curriculum learning. :contentReference[oaicite:23]{index=23}

---

## 12. Future Directions

The report suggests several possible extensions:

- Recurrent policies such as LSTM/GRU for longer memory
- Sim-to-real transfer with domain randomization
- Hierarchical RL for explore/approach/push/unwedge
- Multi-box or multi-agent extensions
- Learned reward shaping via inverse RL
- Go-Explore-style hard exploration methods  

These directions would further improve robustness and generalization in more realistic robotics settings. :contentReference[oaicite:24]{index=24}

---

## 13. Running the Environment

### Manual gameplay
Use `manual_play.py` to control the robot with the keyboard:

| Key | Action |
|---|---|
| `w` | Move forward |
| `a` | Turn left 45° |
| `q` | Turn left 22.5° |
| `e` | Turn right 22.5° |
| `d` | Turn right 45° |

### Evaluation
The evaluation script can be used to run multiple seeds and compute mean/std reward:

```bash
python evaluate.py --agent_file agent_template.py --runs 10 --seed 0 --max_steps 2000 --wall_obstacles