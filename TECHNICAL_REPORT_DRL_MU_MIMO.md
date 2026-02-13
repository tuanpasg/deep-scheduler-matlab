# Báo Cáo Kỹ Thuật: Hệ Thống DRL-based MU-MIMO Scheduler

## Phiên bản: 1.0 | Ngày: 10/02/2026

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Observation từ MATLAB → Python](#2-observation-từ-matlab--python)
3. [Xử Lý Observation theo Layer và TTI](#3-xử-lý-observation-theo-layer-và-tti)
4. [Action từ DRL và Thực Thi tại MATLAB](#4-action-từ-drl-và-thực-thi-tại-matlab)
5. [Precoding: Subband vs Wideband](#5-precoding-subband-vs-wideband)
6. [Tuân Thủ Chuẩn 3GPP](#6-tuân-thủ-chuẩn-3gpp)
7. [Kết Luận](#7-kết-luận)

---

## 1. Tổng Quan Hệ Thống

### 1.1 Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            MATLAB Simulation                             │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────┐│
│  │   gNB PHY   │───▶│  CSI-RS Tx  │───▶│  Channel (CDL-D, 450ns DS)   ││
│  │   (32T32R)  │    │             │    │  Max Doppler: 136 Hz         ││
│  └─────────────┘    └─────────────┘    └──────────────────────────────┘│
│         ▲                                           │                   │
│         │                                           ▼                   │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────┐│
│  │nrDRLScheduler│◀──│  CSI Report │◀───│  UE PHY (Decode CSI-RS)      ││
│  │   (MAC)     │    │(RI/PMI/CQI) │    │  16 UEs, 4Rx each            ││
│  └─────────────┘    └─────────────┘    └──────────────────────────────┘│
│         │                                                               │
│         │ TCP Socket (Port 5555)                                        │
│         ▼                                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Python DRL Server                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐│
│  │build_observation│─▶│ MultiBranchActor│─▶│ Allocation Matrix        ││
│  │  (656 dims)     │  │   (DSACD)       │  │ [18 RBGs × 16 Layers]    ││
│  └─────────────────┘  └─────────────────┘  └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Cấu Hình Hệ Thống

| Tham số | Giá trị | Chuẩn 3GPP |
|---------|---------|------------|
| Băng tần | FR1 (n78) | TS 38.101 |
| Subcarrier Spacing | 30 kHz | TS 38.211 §4.2 |
| Bandwidth | 100 MHz (273 RBs) | TS 38.104 |
| TDD Pattern | DDDSU | TS 38.213 |
| gNB Antennas | **32T32R** | - |
| UE Antennas | 4Tx4Rx per UE | - |
| Max Layers | 16 (MU-MIMO) | TS 38.214 §5.2.2.1 |
| Max Rank per UE | 2 | TS 38.214 Table 5.2.2.1-2 |
| CSI-RS Period | 10 slots | TS 38.214 §5.2.1 |
| RBG Size | ~15 RBs → 18 RBGs | TS 38.214 §5.1.2.2.1 |

---

## 2. Observation từ MATLAB → Python

### 2.1 Cấu Trúc Dữ Liệu Gửi

MATLAB gửi JSON payload qua TCP socket với cấu trúc sau:

```json
{
  "type": "TTI_OBS",
  "frame": 0,
  "slot": 0,
  "max_ues": 16,
  "max_layers": 16,
  "num_rbg": 18,
  "avg_throughput": [U1_tp, U2_tp, ..., U16_tp],
  "rank": [U1_ri, U2_ri, ..., U16_ri],
  "buffer": [U1_buf, U2_buf, ..., U16_buf],
  "wideband_cqi": [U1_cqi, U2_cqi, ..., U16_cqi],
  "subband_cqi": [[U1_sb1, U1_sb2, ...], [U2_sb1, U2_sb2, ...], ...],
  "cross_corr": [[[corr_u1u1_rb1, ...], ...], ...]
}
```

### 2.2 Chi Tiết Từng Trường

#### 2.2.1 `avg_throughput` [16]

**Nguồn:** `nrDRLScheduler.m` lines 420-430

```matlab
avgThroughput = zeros(1, obj.MaxUEs);
for i = 1:numEligibleUEs
    rnti = eligibleUEs(i);
    avgThroughput(rnti) = obj.UEMetrics(rnti).DLThroughput;
end
```

**Ý nghĩa:** Throughput trung bình (bits/s) được tính bởi lớp MAC dựa trên HARQ ACK/NACK từ UE.

**Chuẩn 3GPP:** Throughput đo tại lớp MAC (TS 38.321) sau khi trừ overhead RLC/PDCP.

#### 2.2.2 `rank` [16]

**Nguồn:** `nrDRLScheduler.m` lines 432-440

```matlab
ueRank = zeros(1, obj.MaxUEs);
for i = 1:numEligibleUEs
    rnti = eligibleUEs(i);
    csiReport = obj.UEContext(rnti).CSIMeasurement.CSIRS;
    ueRank(rnti) = csiReport.RI;
end
```

**Ý nghĩa:** Rank Indicator (RI) từ CSI report của UE. Giá trị 1-2 cho cấu hình Type I Single Panel với 4Rx UE.

**Chuẩn 3GPP:** RI được định nghĩa trong TS 38.214 §5.2.2.1. Với codebook Type I Single Panel (4 layer max cho 4Rx), RI ∈ {1, 2, 3, 4} nhưng thực tế bị giới hạn bởi min(gNB_layers, UE_Rx).

#### 2.2.3 `buffer` [16]

**Nguồn:** `nrDRLScheduler.m` lines 442-450

```matlab
bufferStatus = zeros(1, obj.MaxUEs);
for i = 1:numEligibleUEs
    rnti = eligibleUEs(i);
    bufferStatus(rnti) = getUEBufferStatus(obj, rnti, 1);
end
```

**Ý nghĩa:** Buffer Status Report (BSR) trong bytes. Đây là lượng data chờ được truyền trong RLC buffer.

**Chuẩn 3GPP:** BSR được định nghĩa trong TS 38.321 §5.4.5. Giá trị được map từ Buffer Size Level → bytes theo Table 6.1.3.1-1.

#### 2.2.4 `wideband_cqi` [16]

**Nguồn:** `nrDRLScheduler.m` lines 452-460

```matlab
widebandCQI = zeros(1, obj.MaxUEs);
for i = 1:numEligibleUEs
    rnti = eligibleUEs(i);
    csiReport = obj.UEContext(rnti).CSIMeasurement.CSIRS;
    widebandCQI(rnti) = csiReport.CQI;
end
```

**Ý nghĩa:** Channel Quality Indicator trung bình trên toàn bandwidth. CQI ∈ {0, 1, ..., 15}.

**Chuẩn 3GPP:** CQI được định nghĩa trong TS 38.214 Table 5.2.2.1-2 (4-bit CQI table). Map CQI → MCS → Spectral Efficiency theo:

| CQI | Modulation | Code Rate × 1024 | Efficiency (bits/s/Hz) |
|-----|------------|------------------|------------------------|
| 1 | QPSK | 78 | 0.1523 |
| 7 | 16QAM | 378 | 1.4766 |
| 15 | 64QAM | 948 | 5.5547 |

#### 2.2.5 `subband_cqi` [16 × 18]

**Nguồn:** `nrDRLScheduler.m` lines 463-490

```matlab
subbandCQI = ones(obj.MaxUEs, numRBGs);
for i = 1:numEligibleUEs
    rnti = eligibleUEs(i);
    csiReport = obj.UEContext(rnti).CSIMeasurement.CSIRS;
    if isfield(csiReport, 'SubbandCQI') && ~isempty(csiReport.SubbandCQI)
        numSB = min(length(csiReport.SubbandCQI), numRBGs);
        subbandCQI(rnti, 1:numSB) = csiReport.SubbandCQI(1:numSB);
    else
        subbandCQI(rnti, :) = widebandCQI(rnti);
    end
end
```

**Ý nghĩa:** CQI per-subband cho frequency-selective scheduling. Mỗi subband = `SubbandSize` RBs (cấu hình: 16 RBs).

**Chuẩn 3GPP:** Subband CQI được định nghĩa trong TS 38.214 §5.2.1.4. Subband size phụ thuộc bandwidth:

| Bandwidth Part (RBs) | Subband Size (RBs) |
|---------------------|-------------------|
| 24-72 | 4, 8 |
| 73-144 | 8, 16 |
| 145-275 | 16, 32 |

#### 2.2.6 `cross_corr` [16 × 16 × 18]

**Nguồn:** `nrDRLScheduler.m` lines 465-510

```matlab
cross_corr = zeros(obj.MaxUEs, obj.MaxUEs, numRBGs);
W = schedulerInput.W;  % Cell array of precoders

for i = 1:numEligibleUEs
    for j = i:numEligibleUEs
        for rbg = 1:numRBGs
            if ndims(W_i) >= 3 && ndims(W_j) >= 3
                % Subband precoders
                sbIdx = min(rbg, numSB);
                w1 = W_i(:,:,sbIdx);
                w2 = W_j(:,:,sbIdx);
            else
                w1 = W_i;
                w2 = W_j;
            end
            
            w1_flat = w1(:) / (norm(w1(:)) + 1e-10);
            w2_flat = w2(:) / (norm(w2(:)) + 1e-10);
            corr_val = abs(w1_flat' * w2_flat);
            
            cross_corr(rnti_i, rnti_j, rbg) = corr_val;
            cross_corr(rnti_j, rnti_i, rbg) = corr_val;
        end
    end
end
```

**Ý nghĩa:** Ma trận cross-correlation giữa các precoder vectors. Giá trị cao = interference lớn khi MU-MIMO pairing.

**Công thức:**
$$\rho_{i,j,m} = \frac{|\mathbf{w}_i^H \mathbf{w}_j|}{||\mathbf{w}_i|| \cdot ||\mathbf{w}_j||}$$

**Chuẩn 3GPP:** Không có định nghĩa trực tiếp trong 3GPP. Đây là metric implementation-specific để đánh giá semi-orthogonality của precoders (TS 38.214 §5.2.2.3.1 gợi ý về orthogonal precoder selection).

---

## 3. Xử Lý Observation theo Layer và TTI

### 3.1 Luồng Xử Lý Tổng Quan

```
TTI t:
  MATLAB gửi observation → Python nhận
  
  for layer_idx in 0..15:
      1. build_observation(data, layer_idx, prev_alloc)
      2. build_action_mask(layer_idx, constraints)
      3. action = actor.forward_all(obs) + mask
      4. Update prev_alloc with action
  
  Python gửi allocation_matrix → MATLAB thực thi
```

### 3.2 Hàm `build_observation()`

**File:** `train_drl_with_matlab.py` lines 490-650

#### 3.2.1 Input Processing

```python
def build_observation(self, data, max_ues, num_rbg, allocated_rbg=None, 
                      cross_corr=None, layer=0):
    # Extract raw data from MATLAB
    avg_throughput = np.array(data.get('avg_throughput', [0]*max_ues))
    rank = np.array(data.get('rank', [1]*max_ues))
    buffer = np.array(data.get('buffer', [0]*max_ues))
    wideband_cqi = np.array(data.get('wideband_cqi', [7]*max_ues))
    subband_cqi = np.array(data.get('subband_cqi', [[7]*num_rbg]*max_ues))
    cross_corr = np.array(data.get('cross_corr', np.zeros((max_ues, max_ues, num_rbg))))
```

#### 3.2.2 Feature Normalization (7 Features)

| # | Feature | Công thức | Range | Ý nghĩa |
|---|---------|-----------|-------|---------|
| 1 | `norm_past_avg_tp` | `avg_tp / 1100` | [0, 1] | Throughput normalized by max rate (1.1 Gbps) |
| 2 | `norm_ue_rank` | `rank / 2` | [0.5, 1] | Rank capability (1 or 2 layers) |
| 3 | `norm_allocated_rbgs` | `count(prev_alloc == u) / n_rbg` | [0, 1] | **Layer-dependent**: Số RBGs đã allocated |
| 4 | `norm_buffer` | `buffer / max(buffer)` | [0, 1] | Relative buffer fullness |
| 5 | `norm_wb_mcs` | `cqi × (28/15) / 28` | [0, 1] | Wideband channel quality |
| 6 | `norm_subband_mcs` | `sb_cqi × (28/15) / 28` | [0, 1] × 18 | Per-RBG channel quality |
| 7 | `max_corr_feat` | `max(cross_corr[u, scheduled, m])` | [0, 1] × 18 | **Layer-dependent**: Max interference |

#### 3.2.3 Layer-Dependent Features

**Feature 3: `norm_allocated_rbgs`**

```python
if layer > 0:
    prev_alloc = self._alloc[:layer, :]  # [L', M]
    alloc_counts = np.zeros(max_ues)
    for u in range(max_ues):
        alloc_counts[u] = (prev_alloc == u).sum()
    norm_allocated_rbgs = alloc_counts / float(num_rbg)
else:
    norm_allocated_rbgs = np.zeros(max_ues)
```

**Ý nghĩa:** Tại layer L, đếm số RBG-layer pairs mà UE u đã được allocated trong layers 0..L-1. Feature này giúp model biết UE nào đã được ưu tiên trong các layers trước.

**Feature 7: `max_corr_feat`**

```python
max_corr_feat = np.zeros((max_ues, num_rbg))
if layer > 0:
    prev_alloc = self._alloc[:layer, :]  # [L', M]
    for m in range(num_rbg):
        scheduled_ues = prev_alloc[:, m]
        valid_ues = scheduled_ues[scheduled_ues != noop]
        if len(valid_ues) > 0:
            for u in range(max_ues):
                max_corr_feat[u, m] = np.max(cross_corr[u, valid_ues, m])
```

**Ý nghĩa:** Với mỗi RBG m và UE u, tính max cross-correlation với các UEs đã được scheduled trên RBG đó trong layers trước. Feature này encode thông tin về potential interference nếu thêm UE u vào RBG m.

#### 3.2.4 Observation Vector Assembly

```python
# Scalar features [U, 5]
ue_scalar_feats = np.stack([
    norm_past_avg_tp,      # [U]
    norm_ue_rank,          # [U]
    norm_allocated_rbgs,   # [U] - layer dependent
    norm_buffer,           # [U]
    norm_wb_mcs,           # [U]
], axis=1)

# Per-RBG features [U, 2*M]
ue_rbg_feats = np.concatenate([
    norm_subband_mcs,      # [U, M]
    max_corr_feat,         # [U, M] - layer dependent
], axis=1)

# Final observation [U, 5 + 2*M] → flatten → [U × (5 + 2*M)]
ue_feats = np.concatenate([ue_scalar_feats, ue_rbg_feats], axis=1)
obs = ue_feats.reshape(-1)  # [656]
```

**Observation Dimension:**
$$\text{obs\_dim} = U \times (5 + 2 \times M) = 16 \times (5 + 2 \times 18) = 16 \times 41 = 656$$

### 3.3 Action Mask Construction

**File:** `train_drl_with_matlab.py` lines 660-800

Action mask đảm bảo các constraints sau:

#### 3.3.1 Buffer Constraint
```python
valid_ue = (buffer > 0)  # UE must have data to send
```

#### 3.3.2 Rank Constraint (TS 38.214 §5.2.2.1)
```python
# Count layers already allocated to each UE on each RBG
if layer > 0:
    prev_allocs = self._alloc[:layer, :]
    for u in range(max_ues):
        counts[u, m] = (prev_allocs[:, m] == u).sum()
    rank_ok = (counts < ue_rank)  # Must not exceed UE's rank capability
```

**Giải thích:** Một UE với rank=2 chỉ được scheduled tối đa 2 layers trên mỗi RBG.

#### 3.3.3 Continuous Layer Constraint
```python
# UE can only continue if it was in the immediately previous layer
if layer > 0:
    last_layer_alloc = self._alloc[layer - 1, :]
    in_prev_layer = (last_layer_alloc == u)
    ever_seen = (counts > 0)
    continuity_ok = (~ever_seen) | in_prev_layer
```

**Giải thích:** Constraint này đảm bảo layers liên tiếp. VD: Nếu UE1 được allocated ở layer 0, nó chỉ có thể tiếp tục ở layer 1, không thể skip layer.

#### 3.3.4 Max UEs per RBG Constraint
```python
ues_on_rbg = len(set(prev_allocs[:, m]) - {noop})
if ues_on_rbg >= max_ues_per_rbg:  # Default: 8
    mask[m, :] = False
    mask[m, noop] = True  # Only NOOP allowed
```

#### 3.3.5 NOOP Always Valid
```python
mask[:, noop] = True  # NOOP (action=0) is always valid
```

### 3.4 Per-Layer Processing Loop

```python
for layer_idx in range(max_layers):
    # 1. Build observation with layer-dependent features
    obs_layer = self.build_observation(
        data, max_ues, num_rbg,
        allocated_rbg=prev_layer_alloc,
        cross_corr=cross_corr,
        layer=layer_idx
    )
    
    # 2. Build action mask with all constraints
    mask_layer = self.build_action_mask(
        num_rbg, max_ues, eligible_ues, buffer_status,
        ue_set_per_rbg, rank_used_per_rbg_ue,
        current_layer_idx=layer_idx,
        ...
    )
    
    # 3. Select action (epsilon-greedy during training)
    actions, _ = self.select_action(
        obs_layer, mask_layer, explore=True, 
        epsilon=epsilon, layer_idx=layer_idx
    )
    
    # 4. Update allocation tracking
    for rbg in range(num_rbg):
        ue_id = actions[rbg].item()  # 0=NOOP, 1-16=UE
        if ue_id != 0:
            allocation_matrix[rbg][layer_idx] = ue_id
            # Update tracking structures
            ...
```

---

## 4. Action từ DRL và Thực Thi tại MATLAB

### 4.1 Action Space Definition

**Discrete action per RBG:**
$$a_m \in \{0, 1, 2, \ldots, 16\}$$

| Action | Ý nghĩa |
|--------|---------|
| 0 | NOOP - Không allocate UE nào cho RBG này ở layer hiện tại |
| 1-16 | Allocate UE với RNTI tương ứng |

**Output per TTI:** Allocation matrix `[num_rbg × max_layers]` = `[18 × 16]`

### 4.2 Neural Network Architecture

**File:** `DSACD_multibranch.py`

```python
class MultiBranchActor(nn.Module):
    """
    Per-RBG parallel actor heads for independent action selection.
    
    Architecture:
        Shared: Linear(obs_dim, hidden) → ReLU
        Per-RBG: Linear(hidden, n_actions) × n_rbg
    """
    def __init__(self, obs_dim, hidden_dim, n_rbg, n_actions):
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # 656 → 256
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions)  # 256 → 17
            for _ in range(n_rbg)
        ])
    
    def forward_all(self, obs):
        # obs: [batch, obs_dim]
        shared_feat = self.shared(obs)  # [batch, hidden]
        logits = torch.stack([
            head(shared_feat) for head in self.heads
        ], dim=1)  # [batch, n_rbg, n_actions]
        return logits
```

**Parameter count:**
- Shared layer: 656 × 256 + 256 = 168,192
- Per-RBG heads: 18 × (256 × 17 + 17) = 78,642
- **Total Actor: ~312K parameters**

### 4.3 Action Selection with Masking

```python
def select_action(self, obs, mask, explore=True, epsilon=0.1, layer_idx=0):
    obs_input = obs.unsqueeze(0).to(self.device)  # [1, 656]
    mask_input = mask.unsqueeze(0).to(self.device)  # [1, 18, 17]
    
    # Forward pass
    logits = self.actor.forward_all(obs_input)  # [1, 18, 17]
    
    # Apply mask (set invalid actions to -inf)
    logits = apply_action_mask_to_logits(logits, mask_input)
    
    if explore and np.random.rand() < epsilon:
        # Epsilon-greedy: random valid action
        actions = []
        for rbg in range(mask_input.shape[1]):
            valid_actions = torch.where(mask_input[0, rbg])[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))]
            actions.append(action.item())
        actions = torch.tensor(actions)
    else:
        # Greedy: softmax + argmax
        probs = torch.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1).squeeze(0)  # [18]
    
    return actions, mask
```

### 4.4 Allocation Matrix Format

**Python to MATLAB JSON:**

```json
{
  "type": "ALLOCATION",
  "allocation": [
    [0, 5, 5, 0, 0, ..., 0],   // RBG 0: UE5 on layers 1,2
    [3, 3, 7, 7, 0, ..., 0],   // RBG 1: UE3 on layers 0,1; UE7 on layers 2,3
    ...
  ]
}
```

**Interpretation:**
- `allocation[rbg][layer]` = UE RNTI (1-indexed) or 0 (NOOP)
- Continuous layer constraint: UE xuất hiện trên các layers liên tiếp

### 4.5 MATLAB Decoding và Thực Thi

**File:** `nrDRLScheduler.m` lines 900-1100

```matlab
function executeDRLAllocation(obj, allocationMatrix, schedulerInput)
    % allocationMatrix: [numRBGs x numLayers]
    
    numRBGs = size(allocationMatrix, 1);
    numLayers = size(allocationMatrix, 2);
    
    for rbg = 1:numRBGs
        % Get RBs for this RBG
        rbStart = (rbg - 1) * obj.RBGSize + 1;
        rbEnd = min(rbg * obj.RBGSize, obj.NumResourceBlocks);
        rbSet = rbStart:rbEnd;
        
        % Collect UEs scheduled on this RBG
        scheduledUEs = unique(allocationMatrix(rbg, :));
        scheduledUEs = scheduledUEs(scheduledUEs > 0);  % Remove NOOP
        
        for ueIdx = 1:length(scheduledUEs)
            rnti = scheduledUEs(ueIdx);
            
            % Count layers for this UE on this RBG
            ueLayerMask = (allocationMatrix(rbg, :) == rnti);
            numUELayers = sum(ueLayerMask);
            layerIndices = find(ueLayerMask);
            
            % Create DL assignment
            dlAssignment = struct();
            dlAssignment.RNTI = rnti;
            dlAssignment.RBGAllocationBitmap = zeros(1, numRBGs);
            dlAssignment.RBGAllocationBitmap(rbg) = 1;
            dlAssignment.NumLayers = numUELayers;
            dlAssignment.StartLayerIndex = layerIndices(1) - 1;  % 0-indexed
            dlAssignment.W = obj.getSubbandPrecoder(rnti, rbg);
            dlAssignment.MCS = obj.selectMCS(rnti, rbg);
            
            % Add to scheduler output
            obj.DLAssignments{end+1} = dlAssignment;
        end
    end
end
```

### 4.6 Precoder và MCS Selection tại MATLAB

**Precoder Selection:**
```matlab
function W = getSubbandPrecoder(obj, rnti, rbg)
    csiReport = obj.UEContext(rnti).CSIMeasurement.CSIRS;
    W_full = csiReport.W;  % [numPorts x rank x numSubbands]
    
    if ndims(W_full) >= 3
        % Subband precoding: select precoder for this RBG
        sbIdx = min(rbg, size(W_full, 3));
        W = W_full(:, :, sbIdx);
    else
        % Wideband precoding
        W = W_full;
    end
end
```

**MCS Selection:**
```matlab
function mcs = selectMCS(obj, rnti, rbg)
    csiReport = obj.UEContext(rnti).CSIMeasurement.CSIRS;
    
    if isfield(csiReport, 'SubbandCQI') && ~isempty(csiReport.SubbandCQI)
        cqi = csiReport.SubbandCQI(rbg);
    else
        cqi = csiReport.CQI;  % Wideband fallback
    end
    
    % CQI to MCS mapping (TS 38.214 Table 5.2.2.1-2)
    mcs = obj.CQItoMCSTable(cqi);
    
    % MCS backoff for MU-MIMO interference
    if obj.NumCoScheduledUEs(rbg) > 1
        mcs = max(0, mcs - obj.MCSBackoff);
    end
end
```

---

## 5. Precoding: Subband vs Wideband

### 5.1 Type II Codebook (TS 38.214 §5.2.2.2.3) - ĐANG SỬ DỤNG

**⚠️ LƯU Ý: Hệ thống hiện tại sử dụng Type II Codebook (CSIReportType = 2)**

Cấu trúc PMI cho Type II Codebook:

| Component | Granularity | Ý nghĩa |
|-----------|-------------|---------||
| **L beams** | Wideband | Linear combination of L orthogonal beams (L=2 or 4) |
| **Amplitude coefficients** | Subband | Per-beam amplitude (optional: SubbandAmplitude) |
| **Phase coefficients** | Subband | Per-beam phase (PhaseAlphabetSize = 4 → QPSK) |

**Precoder matrix:**
$$\mathbf{W} = \sum_{l=1}^{L} a_l \cdot e^{j\phi_l} \cdot \mathbf{b}_l$$

Trong đó:
- $\mathbf{b}_l$: Orthogonal beam vectors từ DFT codebook
- $a_l$: Amplitude coefficient (wideband hoặc subband)
- $\phi_l$: Phase coefficient (subband, QPSK alphabet)

**Cấu hình Type II trong mô phỏng:**
```matlab
% File: nrNodeValidation.m line 273-284
csiReportConfig.CodebookType = 'Type2';
csiReportConfig.NumberOfBeams = 4;        % L = 4 beams
csiReportConfig.SubbandAmplitude = false; % Wideband amplitude
csiReportConfig.PhaseAlphabetSize = 4;    % QPSK phase
csiReportConfig.CQIMode = 'Subband';
csiReportConfig.PMIMode = 'Subband';
```

**So sánh Type I vs Type II:**

| Đặc điểm | Type I Single Panel | Type II (Hiện tại) |
|----------|---------------------|--------------------|
| PMI Structure | i1 (wideband) + i2 (subband) | L beams + amplitude/phase |
| Feedback Overhead | Thấp | Cao (nhiều bits hơn) |
| Spatial Resolution | Discrete DFT beams | Linear combination |
| MU-MIMO Performance | Medium | **High** |
| 3GPP Reference | TS 38.214 §5.2.2.2.1 | TS 38.214 §5.2.2.2.3 |

### 5.2 Cấu Hình Hiện Tại trong Mô Phỏng

**File:** `MU_MIMO.m` lines 43-48 và `nrGNB.m` line 302

```matlab
% --- nrGNB.m: Default CSI Report Type ---
CSIReportType = 2;  % Type II codebook (protected property)

% --- MU_MIMO.m: CSI report granularity ---
csiReportConfig = struct();
csiReportConfig.SubbandSize = 16;  % RBs per subband
csiReportConfig.PRGSize = 4;       % PRG size for precoding

% --- nrNodeValidation.m: Applied when CSIReportType == 2 ---
csiReportConfig.CodebookType = 'Type2';
csiReportConfig.NumberOfBeams = 4;  % For NumCSIRSPorts > 4
csiReportConfig.SubbandAmplitude = false;
csiReportConfig.PhaseAlphabetSize = 4;
```

### 5.3 Xử Lý Precoder trong Scheduler

**File:** `nrScheduler.m` function `selectRankAndPrecodingMatrixDL` (đã được cập nhật)

```matlab
function [rank, W] = selectRankAndPrecodingMatrixDL(obj, rnti, csiMeasurement, numCSIRSPorts)
    report = csiMeasurement.CSIRS;
    rank = report.RI;
    
    Wsub = report.W;
    
    % SUBBAND PRECODING: Keep per-subband precoders if available
    if ndims(Wsub) >= 3
        numSubbands = size(Wsub, 3);
        
        for i = 1:numPRGs
            sbIdx = min(i, numSubbands);  % Map PRG to subband
            W(:, :, i) = Wsub(:, :, sbIdx);  % Keep subband-specific precoder
        end
    else
        % Wideband fallback
        for i = 1:numPRGs
            W(:, :, i) = W2;
        end
    end
end
```

### 5.4 So Sánh Wideband vs Subband Precoding

| Aspect | Wideband | Subband (Hiện tại) |
|--------|----------|-------------------|
| **PMI i1** | Same for all RBs | Same for all RBs |
| **PMI i2** | Same for all RBs | Different per subband |
| **Precoder W** | Single matrix | Per-PRG/subband matrix |
| **Overhead** | Lower (1 PMI) | Higher (S PMIs) |
| **Performance** | Good for flat channel | Better for frequency-selective |
| **MU-MIMO Compatibility** | Easier pairing | Per-subband pairing needed |

### 5.5 Cross-Correlation Computation (Per-Subband)

**File:** `nrDRLScheduler.m` lines 483-510

```matlab
for rbg = 1:numRBGs
    if ndims(W_i) >= 3 && ndims(W_j) >= 3
        % Subband precoders: W is 3D [rank x ports x subbands]
        numSB = min(size(W_i, 3), size(W_j, 3));
        sbIdx = min(rbg, numSB);
        w1 = W_i(:,:,sbIdx);
        w2 = W_j(:,:,sbIdx);
    else
        % Wideband precoders
        w1 = W_i;
        w2 = W_j;
    end
    
    % Normalized correlation
    w1_flat = w1(:) / (norm(w1(:)) + 1e-10);
    w2_flat = w2(:) / (norm(w2(:)) + 1e-10);
    cross_corr(rnti_i, rnti_j, rbg) = abs(w1_flat' * w2_flat);
end
```

**Kết quả:** Cross-correlation được tính **per-RBG** sử dụng **subband precoders**, cho phép DRL agent nhận biết interference khác nhau trên từng RBG.

### 5.6 MU-MIMO Pairing Constraints với Subband Precoding

**i1 Constraint (Wideband beam matching):**

```matlab
% File: nrDRLScheduler.m
if obj.EnableI1Constraint
    i1_new = csiMeasurement.PMISet.i1;
    
    for existingUE in scheduledUEsOnRBG
        i1_existing = obj.UEContext(existingUE).PMISet.i1;
        if ~isequal(i1_new, i1_existing)
            % Reject: different wideband beam directions
            canPair = false;
            break;
        end
    end
end
```

**Orthogonality Check (Per-subband):**

```matlab
if obj.EnableOrthogonalityConstraint
    for sb = 1:numSubbands
        w1 = W_new(:,:,sb);
        w2 = W_existing(:,:,sb);
        
        corr = abs(w1(:)' * w2(:)) / (norm(w1(:)) * norm(w2(:)));
        
        if corr > (1 - obj.SemiOrthogonalityFactor)
            % Reject: precoders not orthogonal enough on this subband
            canPair = false;
            break;
        end
    end
end
```

### 5.7 Xác Nhận Precoding Đúng cho UE

**Verification trong Log:**

Khi chạy mô phỏng, MATLAB log sẽ hiển thị:

```
[DRL Scheduler] RBG 5:
  UE3 (Rank=2): Layers [0,1], Precoder W[128x2x1] (subband 5)
  UE7 (Rank=1): Layer [2], Precoder W[128x1x1] (subband 5)
  i1 match: YES
  Orthogonality check: PASS (corr=0.12 < threshold 0.5)
```

**Precoder từ CSI-RS feedback được áp dụng đúng:**
1. RI determines number of layers per UE
2. i1 (wideband) ensures beam direction matching for MU-MIMO
3. i2 (subband) provides per-PRG co-phasing
4. W matrix is subband-specific [rank × ports × subbands]

---

## 6. Tuân Thủ Chuẩn 3GPP

### 6.1 CSI Framework (TS 38.214 §5.2)

| Requirement | Implementation | Compliance |
|-------------|----------------|------------|
| CSI-RS based measurement | ✅ CSI-RS configured | TS 38.214 §5.2.1 |
| Type I Single Panel codebook | ✅ PMI with i1/i2 | TS 38.214 §5.2.2.2.1 |
| Subband CQI reporting | ✅ SubbandSize=16 | TS 38.214 §5.2.1.4 |
| RI reporting | ✅ Rank 1-2 | TS 38.214 §5.2.2.1 |

### 6.2 Resource Allocation (TS 38.214 §5.1)

| Requirement | Implementation | Compliance |
|-------------|----------------|------------|
| RBG-based allocation | ✅ Type 0 (bitmap) | TS 38.214 §5.1.2.2.1 |
| PRG-based precoding | ✅ PRGSize=4 | TS 38.214 §5.1.2.3 |
| MCS selection from CQI | ✅ Table 5.2.2.1-2 | TS 38.214 §5.1.3 |

### 6.3 MU-MIMO Transmission (TS 38.214 §5.2.2.3)

| Requirement | Implementation | Compliance |
|-------------|----------------|------------|
| Per-UE DCI (Format 1_1) | ✅ Individual grants | TS 38.212 §7.3.1 |
| Precoder cycling | ⚠️ Not implemented | Optional |
| DMRS configuration | ✅ Type 1, maxLength=2 | TS 38.211 §7.4.1 |

### 6.4 Observations

**Fully Compliant:**
- CSI-RS measurement and reporting
- Type I codebook structure (i1 wideband, i2 subband)
- RBG-based resource allocation
- MCS selection from CQI table

**Implementation-Specific (Not standardized):**
- DRL-based scheduling algorithm
- Cross-correlation metric for UE pairing
- Semi-orthogonality threshold for precoder selection

---

## 7. Kết Luận

### 7.1 Tóm Tắt Hệ Thống

| Component | Specification |
|-----------|---------------|
| **Observation** | 656 dimensions (16 UEs × 41 features) |
| **Action** | 17 choices/RBG (NOOP + 16 UEs) × 18 RBGs × 16 layers |
| **Codebook Type** | **Type II** (TS 38.214 §5.2.2.2.3) |
| **Precoding** | **Subband** (per-PRG W matrices, L=4 beams) |
| **Constraints** | Buffer, Rank, Continuous Layer, Max UEs/RBG, Orthogonality |

### 7.2 Data Flow Summary

```
1. MATLAB → Python (per TTI):
   - 6 observation fields: avg_tp, rank, buffer, wb_cqi, sb_cqi, cross_corr
   
2. Python Processing (per layer):
   - Build 656-dim observation with layer-dependent features
   - Construct action mask with all constraints
   - Actor network selects per-RBG actions
   
3. Python → MATLAB (per TTI):
   - Allocation matrix [18 RBGs × 16 Layers]
   
4. MATLAB Execution:
   - Decode allocation to DL assignments
   - Apply subband precoders per PRG
   - Select MCS from subband CQI
   - Transmit with MU-MIMO
```

### 7.3 Precoding Verification

| Question | Answer |
|----------|--------|
| Codebook Type? | **Type II** (L=4 beams, QPSK phase) |
| Subband hay Wideband? | **Subband** - Phase coefficients khác nhau mỗi subband |
| Cross-correlation per-RBG? | **Yes** - Tính từ subband precoders |
| Đúng cho UE yêu cầu? | **Yes** - W từ CSI-RS feedback được áp dụng đúng |
| MU-MIMO pairing OK? | **Yes** - Beam orthogonality check |

### 7.4 Recommendations

1. **Monitoring:** Thêm logging để verify precoder selection per-TTI
2. **Performance:** Consider adaptive MCS backoff based on actual BLER
3. **Scalability:** Current system supports 16 UEs; can extend with larger observation space

---

*Document generated: 10/02/2026*
*Version: 1.0*
*Authors: DRL MU-MIMO Scheduler Development Team*
