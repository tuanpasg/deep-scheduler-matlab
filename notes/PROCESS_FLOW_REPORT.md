# DRL Training Server Process Flow Report
## 5G NR MU-MIMO Scheduler with Layer-Step MDP

---

## 1. System Overview

This system implements a **Deep Reinforcement Learning (DRL) training server** for 5G NR downlink MU-MIMO scheduling. The architecture follows a **client-server model**:

- **MATLAB (Client)**: Runs 5G simulation, sends observations, applies scheduling decisions
- **Python (Server)**: Runs DSACD algorithm, generates allocation decisions, trains neural networks

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_rbg` | 18 | Resource Block Groups per TTI |
| `max_layers` | 16 | Spatial layers (MIMO streams) |
| `max_ues` | 16 | Maximum UEs in system |
| `max_ue_per_rbg` | 8 | Max distinct UEs per RBG (Table 4) |
| `max_rank_per_ue` | 2 | Max layers per UE per RBG (Table 4) |

---

## 2. Communication Protocol

### 2.1 Message Flow (Per TTI)

```
┌──────────────────┐                    ┌──────────────────┐
│      MATLAB      │                    │      Python      │
│   (Simulator)    │                    │   (DRL Agent)    │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
         │  TTI_OBS (JSON + '\n')               │
         │  - type: "TTI_OBS"                   │
         │  - features: [16 UEs × feat_dim]    │
         │  - eligible_ues: [1, 2, 5, ...]     │
         │  - max_ues, max_layers, num_rbg     │
         │  - (optional) prev_tti_throughput   │
         │───────────────────────────────────→ │
         │                                       │
         │                               1. Store experiences (prev TTI)
         │                               2. Generate allocation (16 layers)
         │                               3. Train networks (if ready)
         │                                       │
         │  TTI_ALLOC (JSON + '\n')             │
         │  - type: "TTI_ALLOC"                 │
         │  - allocation: [18 RBGs × 16 layers] │
         │ ←─────────────────────────────────── │
         │                                       │
         ▼                                       ▼
    Apply allocation                      Wait for next TTI
    (Zero-forcing precoding)
```

### 2.2 Data Structures

**TTI_OBS (MATLAB → Python):**
```json
{
  "type": "TTI_OBS",
  "frame": 0,
  "slot": 5,
  "max_ues": 16,
  "max_layers": 16,
  "num_rbg": 18,
  "eligible_ues": [1, 2, 5, 8, 11],
  "features": [
    [tput_1, rank_1, alloc_ratio_1, buffer_1, wbcqi_1, sbcqi_1, ...],
    [tput_2, rank_2, alloc_ratio_2, buffer_2, wbcqi_2, sbcqi_2, ...],
    ...
  ],
  "prev_tti_throughput": 50000000  // Optional: for reward calculation
}
```

**TTI_ALLOC (Python → MATLAB):**
```json
{
  "type": "TTI_ALLOC",
  "allocation": [
    [3, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // RBG 0
    [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // RBG 1
    ...  // 18 RBGs total, 16 layers each
  ]
}
```
- Value `0` = NO_ALLOC
- Value `1-16` = UE ID (1-indexed for MATLAB)

---

## 3. Layer-Step MDP Architecture

### 3.1 Why Layer-Step (Not TTI-Step)?

The paper designs scheduling as a **sequential per-layer decision** process:

| Approach | Step Definition | Actions per Step | Transitions per TTI |
|----------|-----------------|------------------|---------------------|
| **TTI-Step** (WRONG) | 1 TTI = 1 step | 18×16 = 288 actions | 1 |
| **Layer-Step** (CORRECT) | 1 Layer = 1 step | 18 actions | 16 |

**Layer-Step is correct because:**
1. DSACD multi-branch outputs `[n_rbg]` actions (18 actions per forward pass)
2. Each layer decision affects subsequent layer constraints
3. Agent must learn sequential allocation strategy
4. Paper reward is **incremental gain per layer**

### 3.2 State-Action-Reward Definition

```
Layer-Step MDP for one TTI:
═══════════════════════════════════════════════════════════════════════════

Layer 0:
  State s_{t,0}:  [UE features] with d_u = 0 for all UEs
  Action a_{t,0}: [a_0, a_1, ..., a_17] ∈ {0..16}^18 (UE per RBG or NO_ALLOC)
  Reward r_{t,0}: Incremental throughput from layer 0
  Next:   s_{t,1}

Layer 1:
  State s_{t,1}:  [UE features] with d_u updated (allocated RBGs)
  Action a_{t,1}: [a_0, a_1, ..., a_17] ∈ {0..16}^18
  Reward r_{t,1}: Incremental throughput from layer 1
  Next:   s_{t,2}

...

Layer 15:
  State s_{t,15}: [UE features] with d_u updated (after 14 layers)
  Action a_{t,15}: [a_0, a_1, ..., a_17] ∈ {0..16}^18
  Reward r_{t,15}: Incremental throughput from layer 15
  Next:   s_{t+1,0} (first layer of NEXT TTI)

═══════════════════════════════════════════════════════════════════════════
Total: 16 transitions stored in replay buffer per TTI
```

### 3.3 State (Observation) Update

The observation changes **between layers** due to allocation progress:

```python
# Feature column 2: alloc_ratio (d_u / num_rbg)
# Updated BEFORE each layer's decision

for layer_idx in range(16):
    # Update d_u = number of RBGs where UE u is scheduled
    for u in range(max_ues):
        d_u = sum(allocated_rbg[u])        # Count RBGs with this UE
        features_dyn[u][2] = d_u / num_rbg  # Normalize to [0, 1]
    
    # Build observation WITH updated features
    obs_layer = build_observation(features_dyn)  # Different per layer!
    
    # Select action based on obs_layer
    actions = select_action(obs_layer, mask)
    
    # Apply actions (updates allocated_rbg, ue_set, rank_used)
```

**Key insight**: `obs_layer[0]` ≠ `obs_layer[1]` ≠ ... ≠ `obs_layer[15]`

---

## 4. Detailed TTI Processing Flow

### 4.1 Main Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │       MATLAB sends TTI_OBS         │
                    │  (features, eligible_ues, etc.)    │
                    └───────────────┬─────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    handle_tti_observation()                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  IF prev_obs_layers != None:                                        │  │
│  │    • Get prev_allocation_matrix (from LAST TTI)                     │  │
│  │    • Calculate layer_rewards[16] using prev_allocation              │  │
│  │    • For layer in 0..15:                                            │  │
│  │        - state = prev_obs_layers[layer]                             │  │
│  │        - action = prev_actions_layers[layer]                        │  │
│  │        - reward = layer_rewards[layer]                              │  │
│  │        - next_state = prev_obs_layers[layer+1] OR curr_first_obs    │  │
│  │        - replay_buffer.push(state, action, reward, next_state)      │  │
│  │        - layer_steps += 1                                           │  │
│  │    • Maybe train (every train_freq layer-steps)                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         process_tti()                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. total_steps += 1 (TTI counter)                                  │  │
│  │  2. Build initial observation                                       │  │
│  │  3. Initialize networks if first TTI                                │  │
│  │  4. generate_allocation_matrix() → [18×16] + obs_layers + actions   │  │
│  │  5. Send allocation to MATLAB                                       │  │
│  │  6. Shift tracking: prev_allocation ← last_allocation               │  │
│  │                     last_allocation ← new allocation                │  │
│  │  7. Return (all_obs_layers, all_actions) for next iteration        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │    Python sends TTI_ALLOC          │
                    │  allocation: [18 RBGs × 16 layers] │
                    └─────────────────────────────────────┘
```

### 4.2 generate_allocation_matrix() - Core Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER-BY-LAYER ALLOCATION GENERATION                     │
├─────────────────────────────────────────────────────────────────────────────┤

Initialize trackers:
  - rank_used_per_rbg_ue[18][16] = 0      # How many layers UE has on each RBG
  - ue_set_per_rbg[18] = {}               # Set of UEs on each RBG
  - allocated_rbg[16][18] = 0             # Binary: UE scheduled on RBG?
  - allocation_matrix[18][16] = 0         # Output: UE ID per RBG per layer

FOR layer_idx = 0 to 15:
  │
  ├─► Step 1: Update features_dyn (alloc_ratio column)
  │     for u in range(max_ues):
  │         d_u = sum(allocated_rbg[u])   # RBGs allocated to UE u
  │         features_dyn[u][2] = d_u / 18  # Normalize
  │
  ├─► Step 2: Build obs_layer from features_dyn
  │     obs_layer = build_observation(features_dyn)
  │     all_obs_layers.append(obs_layer)
  │
  ├─► Step 3: Build action mask (Table 4 constraints)
  │     mask[rbg][action] = True/False
  │     
  │     For each RBG:
  │       - NO_ALLOC (action=16) always valid
  │       - UE action valid IF:
  │           • UE is eligible
  │           • UE has buffer > 0
  │           • IF UE already on RBG: rank_used < 2
  │           • IF UE new to RBG: distinct_count < 8
  │
  ├─► Step 4: Select actions using Actor network
  │     actions = actor.forward_all(obs_layer)  # [18] actions
  │     Apply epsilon-greedy exploration
  │     all_actions.append(actions)
  │
  └─► Step 5: Apply actions, update trackers
        for rbg in range(18):
          action = actions[rbg]
          if action == NO_ALLOC: continue
          
          ue0 = action                # 0-based
          ue_id = ue0 + 1             # 1-based for MATLAB
          
          # Validate constraints (safety check)
          if valid:
            allocation_matrix[rbg][layer_idx] = ue_id
            rank_used_per_rbg_ue[rbg][ue0] += 1
            ue_set_per_rbg[rbg].add(ue0)
            allocated_rbg[ue0][rbg] = 1

RETURN allocation_matrix, all_actions, all_masks, all_obs_layers
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Experience Storage Flow

### 5.1 Timing Diagram

```
Time ──────────────────────────────────────────────────────────────────────────►

TTI (t-1)                      TTI (t)                        TTI (t+1)
───────────────────────────────────────────────────────────────────────────────

MATLAB:
  [Execute allocation (t-1)]    [Send OBS(t)]                 [Send OBS(t+1)]
         │                           │                             │
         │                           │                             │
Python:  │                           │                             │
         │                    [Receive OBS(t)]               [Receive OBS(t+1)]
         │                           │                             │
         │                    Store experiences               Store experiences
         │                    for TTI (t-1):                  for TTI (t):
         │                      - obs from (t-1)                - obs from (t)
         │                      - actions from (t-1)            - actions from (t)
         │                      - rewards from (t-1)            - rewards from (t)
         │                      - next_obs = OBS(t)[0]          - next_obs = OBS(t+1)[0]
         │                           │                             │
         │                    [Generate alloc(t)]            [Generate alloc(t+1)]
         │                    [Send to MATLAB]               [Send to MATLAB]
         │                           │                             │
         ▼                           ▼                             ▼
```

### 5.2 Reward Alignment

**Critical**: Rewards must match the allocation that produced them.

```python
# When receiving TTI_OBS for TTI t:
#   - prev_obs_layers    = observations from TTI (t-1)
#   - prev_actions_layers = actions from TTI (t-1)  
#   - last_allocation    = allocation from TTI (t-1) ← USE THIS FOR REWARDS
#   - data               = TTI t observation (may contain metrics for t-1)

# Reward calculation uses LAST allocation (not prev)
prev_alloc = self.last_allocation_matrix  # Allocation that was just executed

# Calculate per-layer rewards based on TTI (t-1) allocation
layer_rewards = calculate_layer_rewards(data, prev_alloc, 16)

# Store transitions for TTI (t-1)
for layer in range(16):
    replay_buffer.push(
        state=prev_obs_layers[layer],       # s_{t-1,l}
        action=prev_actions_layers[layer],  # a_{t-1,l}
        reward=layer_rewards[layer],        # r_{t-1,l}
        next_state=...                       # s_{t-1,l+1} or s_{t,0}
    )
```

---

## 6. Constraint Enforcement (Table 4)

### 6.1 Paper Constraints

| Constraint | Symbol | Value | Meaning |
|------------|--------|-------|---------|
| Max distinct UEs per RBG | \|L\| | 8 | At most 8 different UEs can share one RBG |
| Max rank per UE per RBG | rank | 2 | Each UE can have at most 2 layers on one RBG |
| Max total layers per RBG | - | 16 | Sum of all UE layers ≤ 16 |

### 6.2 Constraint Tracking

```python
# Per-RBG trackers
ue_set_per_rbg[rbg] = set()        # Distinct UEs on this RBG
rank_used_per_rbg_ue[rbg][ue] = 0  # Layers used by UE on this RBG
layers_used_per_rbg[rbg] = 0       # Total layers used on this RBG

# Example state after some allocations:
# RBG 0: UEs {1, 3, 5} with ranks {2, 1, 2} → 3 distinct, 5 total layers
# RBG 1: UEs {2, 4, 6, 7, 8, 9, 10, 11} with all rank 1 → 8 distinct (FULL)
```

### 6.3 Mask Building Logic

```python
def build_action_mask(rbg, ue):
    if ue == NO_ALLOC:
        return True  # Always valid
    
    if ue not in eligible_ues:
        return False
    
    if buffer_status[ue] <= 0:
        return False
    
    if ue in ue_set_per_rbg[rbg]:
        # UE already on this RBG
        return rank_used_per_rbg_ue[rbg][ue] < 2  # Can add 2nd layer?
    else:
        # New UE
        return len(ue_set_per_rbg[rbg]) < 8  # Room for new UE?
```

---

## 7. Training Flow

### 7.1 Training Frequency

```python
# Training happens every train_freq LAYER-STEPS (not TTI-steps)

layer_steps = 0  # Global counter

# After storing 16 transitions per TTI:
layer_steps += 16

# Train condition:
if len(replay_buffer) >= learning_starts:
    if layer_steps % train_freq == 0:
        train_step(batch_size=256)
```

### 7.2 Epsilon Decay

```python
# Epsilon-greedy exploration
# Decay based on LAYER steps (more granular)

epsilon = max(0.1, 1.0 - layer_steps / 50000)

# Examples:
#   layer_steps = 0      → epsilon = 1.0 (full exploration)
#   layer_steps = 25000  → epsilon = 0.5 (50% exploration)
#   layer_steps = 50000+ → epsilon = 0.1 (mostly greedy)
```

### 7.3 Network Update

```python
def train_step(batch_size):
    # Sample from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # shapes:
    #   states:      [batch_size, obs_dim]
    #   actions:     [batch_size, n_rbg] = [256, 18]
    #   rewards:     [batch_size]
    #   next_states: [batch_size, obs_dim]
    
    # DSACD update (actor + dual critics + temperature)
    metrics = updater.update(states, actions, rewards, next_states, dones)
```

---

## 8. Summary Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE TTI PROCESSING FLOW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐                                                              │
│  │ MATLAB     │ ──TTI_OBS──►  ┌──────────────────────────────────────────┐  │
│  │ Simulator  │               │ handle_tti_observation()                 │  │
│  └────────────┘               │   └─► Store 16 layer-step transitions    │  │
│                               │   └─► Maybe train network               │  │
│                               └────────────────┬─────────────────────────┘  │
│                                                │                            │
│                                                ▼                            │
│                               ┌──────────────────────────────────────────┐  │
│                               │ process_tti()                            │  │
│                               │   └─► total_steps++                      │  │
│                               │   └─► Build initial observation          │  │
│                               └────────────────┬─────────────────────────┘  │
│                                                │                            │
│                                                ▼                            │
│                               ┌──────────────────────────────────────────┐  │
│                               │ generate_allocation_matrix()             │  │
│                               │   ┌─────────────────────────────────┐   │  │
│                               │   │ FOR layer = 0 to 15:            │   │  │
│                               │   │   1. Update d_u in features    │   │  │
│                               │   │   2. Build obs_layer           │   │  │
│                               │   │   3. Build action mask         │   │  │
│                               │   │   4. Actor selects actions     │   │  │
│                               │   │   5. Apply + update trackers   │   │  │
│                               │   └─────────────────────────────────┘   │  │
│                               │   └─► Returns: allocation[18][16]       │  │
│                               │                obs_layers[16]           │  │
│                               │                actions_layers[16]       │  │
│                               └────────────────┬─────────────────────────┘  │
│                                                │                            │
│                                                ▼                            │
│  ┌────────────┐               ┌──────────────────────────────────────────┐  │
│  │ MATLAB     │ ◄──TTI_ALLOC──│ send_allocation()                        │  │
│  │ Simulator  │               │   └─► allocation: [18 RBGs × 16 layers]  │  │
│  └────────────┘               └──────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Implementation Details

### 9.1 Counter Meanings

| Counter | Increments | Use |
|---------|------------|-----|
| `total_steps` | +1 per TTI | Logging, checkpoint naming |
| `layer_steps` | +16 per TTI (+1 per layer transition) | Training frequency, epsilon decay |
| `episode_count` | When episode ends | Long-term metrics |

### 9.2 Allocation Matrix Tracking

```python
# Two matrices track allocation history:

prev_allocation_matrix  # Allocation from TTI (t-2), used for nothing now
last_allocation_matrix  # Allocation from TTI (t-1), used for REWARD calculation

# Update sequence in process_tti():
self.prev_allocation_matrix = self.last_allocation_matrix  # Shift
self.last_allocation_matrix = allocation_matrix            # Store new
```

### 9.3 Action Indexing

```
Python action space: [0, 1, 2, ..., max_ues-1, max_ues]
                      └─────── UEs ──────────┘  └─ NO_ALLOC

MATLAB UE IDs:        [1, 2, 3, ..., max_ues, 0]
                      └─────── UEs ─────────┘ └─ NO_ALLOC

Conversion:
  - Python action i (0 ≤ i < max_ues) → MATLAB UE ID (i+1)
  - Python action max_ues             → MATLAB 0 (NO_ALLOC)
```

---

## 10. Conclusion

This implementation correctly follows the **paper's layer-step MDP** approach:

1. **16 separate MDP steps per TTI** (not 1 step with 288 actions)
2. **State updates between layers** (d_u feature reflects allocation progress)
3. **Constraint enforcement** via action masking (Table 4)
4. **Proper reward alignment** (rewards match the allocation that produced them)
5. **DSACD multi-branch** compatible (18 actions per forward pass)

The system is ready for training with proper MATLAB reward feedback.
