"""
MatlabEnvAdapter
================
Drop-in replacement for DeterministicToy5GEnvAdapter.
Wraps TCP communication with MATLAB (nrDRLScheduler.m, TrainingMode=true).

MATLAB gửi toàn bộ raw channel state 1 lần / TTI (TTI_RAW_DATA).
Python tự tính OBS, mask, reward bên trong — logic giống hệt toy5g.
Python chỉ gửi lại allocation matrix (TTI_ALLOC) sau khi loop qua hết các layer.

Protocol:
  MATLAB → Python : TTI_RAW_DATA  {tti, n_layers, n_rbg, max_ues,
                                    eligible_ues, avg_tp_bps, rank,
                                    buffers, wb_cqi, sb_cqi[n_elig x n_rbg],
                                    kappa[n_elig x n_elig]}
  Python → MATLAB : TTI_ALLOC     {alloc_matrix[n_rbg x n_layers],
                                    invalid_count, total_count}
  MATLAB → Python : TTI_DONE      {metrics}

obs_dim  = (5 + 2*n_rbg) * max_sched_ue   (matches toy5g _build_obs)
act_dim  = max_sched_ue + 1               (matches toy5g act_dim)
noop     = max_sched_ue                   (matches toy5g noop)
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional

import numpy as np
import torch

from toy5g_env_adapter import tbs_38214_bytes   # reuse 3GPP TBS helper


# CQI → MCS index mapping (3GPP TS 38.214 Table 5.2.2.1-2)
CQI_TO_MCS = [0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28]


# ─── TCP helpers (identical to old adapter) ───────────────────────────────────

_NL = b"\n"


def _recv_json(sock: socket.socket, rxbuf: bytearray) -> dict:
    while True:
        idx = rxbuf.find(_NL)
        if idx != -1:
            line = bytes(rxbuf[:idx])
            del rxbuf[:idx + 1]
            if not line.strip():
                continue
            return json.loads(line.decode("utf-8"))
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("[MatlabEnv] MATLAB disconnected unexpectedly")
        rxbuf.extend(chunk)


def _send_json(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)


# ─── LayerContext (same as toy5g) ─────────────────────────────────────────────

@dataclass
class LayerContext:
    layer: int
    obs: torch.Tensor        # [obs_dim]
    masks_rbg: torch.Tensor  # [n_rbg, act_dim] bool


# ─── Adapter ──────────────────────────────────────────────────────────────────

class MatlabEnvAdapter:
    """
    Wraps MATLAB simulation as an RL environment with the SAME interface
    as DeterministicToy5GEnvAdapter.

    Obs / mask / reward are computed INTERNALLY in Python (toy5g logic).
    MATLAB only provides raw channel state once per TTI.

    Key attributes (mirrors toy5g):
        max_sched_ue : int
        n_ue         : int   (alias)
        n_layers     : int
        n_rbg        : int
        obs_dim      : int   = (5 + 2*n_rbg) * max_sched_ue
        act_dim      : int   = max_sched_ue + 1
        noop         : int   = max_sched_ue
        eps          : float
    """

    # toy5g constants (kept identical so a model trained on toy5g transfers)
    MAX_RATE_MBPS = 1100.0   # normalisation for avg throughput feature
    MAX_UE_RANK   = 2        # normalisation for rank feature
    MAX_MCS       = 28       # normalisation for CQI/MCS feature
    PRB_PER_RBG   = 18       # PRBs per RBG (default, overridden by MATLAB)
    N_SYMB        = 14       # OFDM symbols per slot
    OVERHEAD      = 18       # RE overhead per PRB

    def __init__(
        self,
        *,
        max_sched_ue: int = 16,
        n_layers: int = 16,
        n_rbg: int = 18,
        port: int = 5555,
        host: str = "0.0.0.0",
        eps: float = 1e-6,
        verbose: bool = True,
    ):
        self.max_sched_ue = max_sched_ue
        self.n_ue         = max_sched_ue
        self.n_layers     = n_layers
        self.n_rbg        = n_rbg
        self.noop         = max_sched_ue
        self.act_dim      = max_sched_ue + 1
        self.obs_dim      = (5 + 2 * n_rbg) * max_sched_ue
        self.eps          = float(eps)
        self.verbose      = verbose

        # ── Per-TTI channel state (populated in begin_tti) ────────────────
        self._eligible_ues: List[int] = []        # RNTI values (1-based), len = numEligible
        self._avg_tp:  np.ndarray = np.zeros(max_sched_ue)   # [U] Mbps
        self._rank:    np.ndarray = np.ones(max_sched_ue)    # [U] RI
        self._buffers: np.ndarray = np.zeros(max_sched_ue)   # [U] bytes
        self._mcs_wb:  np.ndarray = np.zeros(max_sched_ue, dtype=int)   # [U] MCS 0-28
        self._mcs_sb:  np.ndarray = np.zeros((max_sched_ue, n_rbg), dtype=int)  # [U,M]
        self._kappa:   np.ndarray = np.zeros((max_sched_ue, max_sched_ue))  # [U,U]

        # ── Per-TTI allocation (updated in apply_layer_actions) ───────────
        self._alloc = torch.full(
            (n_layers, n_rbg), self.noop, dtype=torch.long
        )
        self._last_transitions: List[Dict] = []
        self._cur_layer: Optional[int] = None
        self._last_metrics: dict = {}

        # ── TCP server ────────────────────────────────────────────────────
        self._rxbuf = bytearray()
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((host, port))
        self._server_sock.listen(1)
        if self.verbose:
            print(f"[MatlabEnv] Listening on {host}:{port}  "
                  f"obs_dim={self.obs_dim}  act_dim={self.act_dim}  "
                  f"n_layers={self.n_layers}  n_rbg={self.n_rbg}")
            print("[MatlabEnv] Waiting for MATLAB to connect...")
        self._client_sock, addr = self._server_sock.accept()
        if self.verbose:
            print(f"[MatlabEnv] MATLAB connected from {addr}")

    # ── Public API (identical to toy5g) ───────────────────────────────────────

    def reset(self):
        """Reset internal state (call once before the training loop)."""
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer  = None
        self._last_metrics = {}

    def begin_tti(self):
        """
        Wait for TTI_RAW_DATA from MATLAB.
        Parse raw channel state into internal arrays (mirrors toy5g.begin_tti).
        """
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None

        msg = self._recv()
        if msg.get("type") != "TTI_RAW_DATA":
            raise RuntimeError(
                f"[MatlabEnv] Expected TTI_RAW_DATA, got: {msg.get('type')}"
            )

        n_elig = len(msg["eligible_ues"])
        U      = self.max_sched_ue
        M      = self.n_rbg

        # Validate geometry
        if msg["n_layers"] != self.n_layers:
            raise RuntimeError(
                f"[MatlabEnv] n_layers mismatch: MATLAB={msg['n_layers']}, "
                f"Python={self.n_layers}"
            )
        if msg["n_rbg"] != M:
            raise RuntimeError(
                f"[MatlabEnv] n_rbg mismatch: MATLAB={msg['n_rbg']}, "
                f"Python={M}"
            )

        self._eligible_ues = [int(x) for x in msg["eligible_ues"]]

        # ── Parse raw arrays (indexed by local position, pad to max_sched_ue) ─
        avg_tp_bps = np.array(msg["avg_tp_bps"], dtype=np.float64)   # [n_elig] bps
        rank_raw   = np.array(msg["rank"],       dtype=np.float64)   # [n_elig] RI
        buf_raw    = np.array(msg["buffers"],    dtype=np.float64)   # [n_elig] bytes
        wb_cqi_raw = np.array(msg["wb_cqi"],     dtype=np.int32)     # [n_elig] 0-15
        sb_cqi_raw = np.array(msg["sb_cqi"],     dtype=np.int32)     # [n_elig x M]
        kappa_raw  = np.array(msg["kappa"],      dtype=np.float64)   # [n_elig x n_elig]

        # Pad/truncate to max_sched_ue
        n_use = min(n_elig, U)

        self._avg_tp  = np.zeros(U)
        self._rank    = np.ones(U)
        self._buffers = np.zeros(U)
        self._mcs_wb  = np.zeros(U, dtype=int)
        self._mcs_sb  = np.zeros((U, M), dtype=int)
        self._kappa   = np.zeros((U, U))

        self._avg_tp[:n_use]     = avg_tp_bps[:n_use] / 1e6          # bps → Mbps
        self._rank[:n_use]       = rank_raw[:n_use]
        self._buffers[:n_use]    = buf_raw[:n_use]
        self._mcs_wb[:n_use]     = [CQI_TO_MCS[min(int(c), 15)] for c in wb_cqi_raw[:n_use]]
        self._mcs_sb[:n_use, :]  = np.vectorize(lambda c: CQI_TO_MCS[min(int(c), 15)])(
            sb_cqi_raw[:n_use, :M]
        )
        self._kappa[:n_use, :n_use] = kappa_raw[:n_use, :n_use]

        if self.verbose:
            tti = msg.get("tti", "?")
            print(f"\n[MatlabEnv] {'='*60}")
            print(f"[MatlabEnv] TTI {tti}")
            print(f"[MatlabEnv] {'='*60}")
            print(f"[MatlabEnv] Raw data received from MATLAB:")
            print(f"[MatlabEnv]   eligible_ues : {self._eligible_ues}  (n={n_elig}, used={n_use})")
            print(f"[MatlabEnv]   avg_tp       : {np.round(self._avg_tp[:n_use], 2).tolist()} [Mbps]")
            print(f"[MatlabEnv]   rank         : {self._rank[:n_use].astype(int).tolist()}")
            print(f"[MatlabEnv]   buffers      : {np.round(self._buffers[:n_use]).astype(int).tolist()} [bytes]")
            print(f"[MatlabEnv]   wb_cqi       : {wb_cqi_raw[:n_use].tolist()}  → mcs_wb: {self._mcs_wb[:n_use].tolist()}")
            print(f"[MatlabEnv]   sb_cqi shape : {sb_cqi_raw.shape}  → mcs_sb shape: {self._mcs_sb.shape}")
            print(f"[MatlabEnv]   kappa shape  : {kappa_raw.shape}  max={float(kappa_raw.max()):.3f}  mean={float(kappa_raw.mean()):.3f}")
            print(f"[MatlabEnv] Dimensions check:")
            print(f"[MatlabEnv]   obs_dim  = (5 + 2×{M}) × {U} = {self.obs_dim}")
            print(f"[MatlabEnv]   act_dim  = {U} + 1 = {self.act_dim}")
            print(f"[MatlabEnv]   noop     = {self.noop}")

    def layer_iter(self) -> Iterator[LayerContext]:
        """
        Yield one LayerContext per spatial layer — computed LOCALLY (no TCP).
        Mirrors toy5g layer_iter.
        """
        for l in range(self.n_layers):
            self._cur_layer = l
            obs   = self._build_obs(layer=l)
            masks = self._build_masks(layer=l)

            if self.verbose:
                valid_per_rbg = masks[:, :-1].sum(dim=1).tolist()   # exclude noop col
                n_valid_total = sum(valid_per_rbg)
                print(f"[MatlabEnv]   Layer {l:2d} | "
                      f"obs.shape={tuple(obs.shape)}  "
                      f"masks.shape={tuple(masks.shape)}  "
                      f"valid_UEs/RBG={valid_per_rbg}  "
                      f"total_valid={n_valid_total}")

            yield LayerContext(layer=l, obs=obs, masks_rbg=masks)

    def apply_layer_actions(
        self, layer_ctx: LayerContext, actions_rbg: torch.Tensor
    ):
        """
        Store actions in _alloc — no TCP.
        actions_rbg : Tensor[n_rbg] int64, values in [0..act_dim-1]
        """
        l = layer_ctx.layer
        a = actions_rbg.cpu().long().clamp(0, self.act_dim - 1)
        self._alloc[l, :] = a

        if self.verbose:
            a_list   = a.tolist()
            n_noop   = sum(1 for x in a_list if x == self.noop)
            n_alloc  = len(a_list) - n_noop
            print(f"[MatlabEnv]   Layer {l:2d} actions | "
                  f"shape={tuple(a.shape)}  "
                  f"allocated={n_alloc}/{len(a_list)} RBGs  "
                  f"noop={n_noop}  "
                  f"values={a_list}")

    def compute_layer_transitions(
        self, layer_ctx: LayerContext
    ) -> List[Dict]:
        """
        Compute rewards and package transitions — locally (no TCP).
        Mirrors toy5g _compute_layer_transitions.
        """
        layer = layer_ctx.layer
        obs   = layer_ctx.obs
        masks = layer_ctx.masks_rbg

        # Reward (toy5g new_reward_compute)
        rewards_m, _ = self._new_reward_compute(layer, masks)

        # Next obs / masks
        next_l     = min(layer + 1, self.n_layers - 1)
        next_obs   = self._build_obs(layer=next_l)
        next_masks = self._build_masks(layer=next_l)

        if self.verbose:
            rw = [round(float(rewards_m[m].item()), 3) for m in range(self.n_rbg)]
            print(f"[MatlabEnv]   Layer {layer:2d} rewards | "
                  f"mean={sum(rw)/len(rw):.3f}  "
                  f"min={min(rw):.3f}  max={max(rw):.3f}  "
                  f"values={rw}")

        out: List[Dict] = []
        for m in range(self.n_rbg):
            tr = {
                "observation":      obs.cpu(),
                "next_observation": next_obs.cpu(),
                "rbg_index":        torch.tensor(m, dtype=torch.long),
                "action":           torch.tensor(
                                        int(self._alloc[layer, m].item()),
                                        dtype=torch.long),
                "reward":           torch.tensor(
                                        float(rewards_m[m].item()),
                                        dtype=torch.float32),
                "action_mask":      masks[m].cpu(),
                "next_action_mask": next_masks[m].cpu(),
            }
            out.append(tr)
            self._last_transitions.append(tr)
        return out

    def finish_tti(self):
        """
        Send TTI_ALLOC to MATLAB, wait for TTI_DONE.
        Mirrors toy5g finish_tti (state update is done by MATLAB simulation).
        """
        # Count invalid / total actions for MATLAB metrics
        alloc_np = self._alloc.cpu().numpy()   # [n_layers, n_rbg]
        n_elig   = len(self._eligible_ues)
        invalid_count = 0
        total_count   = 0

        # Rebuild masks per layer to count invalids
        for l in range(self.n_layers):
            masks_l = self._build_masks(layer=l)
            for m in range(self.n_rbg):
                a = int(alloc_np[l, m])
                total_count += 1
                if a != self.noop and not bool(masks_l[m, a].item()):
                    invalid_count += 1

        # alloc_matrix sent as [n_rbg x n_layers] (MATLAB convention)
        alloc_matlab = alloc_np.T.tolist()   # [n_rbg x n_layers]

        if self.verbose:
            alloc_np_t = self._alloc.cpu().numpy()   # [n_layers, n_rbg]
            n_alloc_total = int((alloc_np_t != self.noop).sum())
            n_noop_total  = int((alloc_np_t == self.noop).sum())
            print(f"[MatlabEnv] Sending TTI_ALLOC:")
            print(f"[MatlabEnv]   alloc_matrix shape : [{self.n_rbg} × {self.n_layers}] (n_rbg × n_layers)")
            print(f"[MatlabEnv]   allocated slots    : {n_alloc_total}  noop slots: {n_noop_total}")
            print(f"[MatlabEnv]   invalid_count={invalid_count}  total_count={total_count}  "
                  f"invalid_rate={invalid_count/max(total_count,1):.3f}")

        self._send({
            "type":          "TTI_ALLOC",
            "alloc_matrix":  alloc_matlab,
            "invalid_count": invalid_count,
            "total_count":   total_count,
        })

        # Wait for TTI_DONE + metrics
        msg = self._recv()
        if msg.get("type") != "TTI_DONE":
            raise RuntimeError(
                f"[MatlabEnv] Expected TTI_DONE, got: {msg.get('type')}"
            )
        self._last_metrics = msg.get("metrics", {})

        if self.verbose:
            m = self._last_metrics
            print(f"[MatlabEnv] TTI_DONE metrics from MATLAB:")
            print(f"[MatlabEnv]   cell_tput={m.get('total_cell_tput', 0):.2f} Mbps  "
                  f"jain={m.get('jain_throughput', 0):.3f}  "
                  f"pf_utility={m.get('pf_utility', 0):.2f}")
            print(f"[MatlabEnv]   layers/RBG={m.get('avg_layers_per_rbg', 0):.2f}  "
                  f"noop_rate={m.get('no_schedule_rate', 0):.3f}  "
                  f"invalid_rate={m.get('invalid_action_rate', 0):.3f}")

    def export_branch_transitions(self) -> List[Dict]:
        return self._last_transitions

    def close(self):
        try:
            self._client_sock.close()
        except Exception:
            pass
        try:
            self._server_sock.close()
        except Exception:
            pass
        if self.verbose:
            print("[MatlabEnv] Connection closed.")

    # ── Internal: OBS (mirrors toy5g _build_obs) ─────────────────────────────

    def _build_obs(self, layer: int) -> torch.Tensor:
        U = self.max_sched_ue
        M = self.n_rbg

        # 1. Normalised avg throughput [U]  (toy5g: avg_tp / max_rate)
        norm_avg_tp = np.clip(self._avg_tp / self.MAX_RATE_MBPS, 0.0, 1.0)

        # 2. Normalised rank [U]  (toy5g: rank / max_ue_rank)
        norm_rank = np.clip(self._rank / float(self.MAX_UE_RANK), 0.0, 1.0)

        # 3. Normalised already-allocated RBGs for this TTI [U]
        #    (toy5g: count of alloc[:layer, :] == u per UE, divided by n_rbg)
        alloc_counts = np.zeros(U)
        if layer > 0:
            prev = self._alloc[:layer, :].cpu().numpy()   # [layer, n_rbg]
            for u in range(U):
                alloc_counts[u] = float(np.sum(prev == u))
        norm_alloc = alloc_counts / max(float(M), 1.0)

        # 4. Normalised buffer [U]  (toy5g: buf / max(buf))
        max_buf = max(float(self._buffers.max()), self.eps)
        norm_buf = self._buffers / max_buf

        # 5. Normalised wideband MCS [U]  (toy5g: mcs / max_mcs)
        norm_wb = self._mcs_wb.astype(float) / float(self.MAX_MCS)

        # 6. Normalised subband MCS [U x M]  (toy5g: curr_mcs_subband / max_mcs)
        norm_sb = self._mcs_sb.astype(float) / float(self.MAX_MCS)

        # 7. Max cross-correlation feature [U x M]
        #    For each (u, m): max kappa(u, v) over UEs v already scheduled on RBG m
        #    in previous layers — uses precomputed wideband kappa matrix.
        max_corr = np.zeros((U, M))
        if layer > 0:
            prev = self._alloc[:layer, :].cpu().numpy()   # [layer, n_rbg]
            for m in range(M):
                scheduled_local = prev[:, m]
                valid = scheduled_local[scheduled_local != self.noop]
                valid = valid[valid < U].astype(int)
                if len(valid) > 0:
                    # kappa[:, valid] — rows = all candidates, cols = scheduled
                    sub = self._kappa[:, valid]            # [U, k]
                    max_corr[:, m] = sub.max(axis=1)

        # Assemble per-UE feature matrix [U x (5 + 2*M)]  (toy5g ue_feats)
        base = np.stack(
            [norm_avg_tp, norm_rank, norm_alloc, norm_buf, norm_wb], axis=1
        )  # [U, 5]
        ue_mat = np.concatenate([base, norm_sb, max_corr], axis=1)  # [U, 5+2M]

        # Flatten column-major (same as toy5g reshape(-1))
        core = torch.tensor(ue_mat.reshape(-1), dtype=torch.float32)

        if core.numel() >= self.obs_dim:
            return core[:self.obs_dim].clone()
        out = torch.zeros(self.obs_dim, dtype=torch.float32)
        out[:core.numel()] = core
        return out

    # ── Internal: Masks (mirrors toy5g _build_masks) ─────────────────────────

    def _build_masks(self, layer: int) -> torch.Tensor:
        U = self.max_sched_ue
        M = self.n_rbg

        # Base: UE valid if buffer > 0
        buf_ok = torch.tensor(self._buffers > 0.0, dtype=torch.bool)  # [U]
        valid_ue = buf_ok.unsqueeze(0).expand(M, -1)                   # [M, U]

        if layer > 0:
            prev = self._alloc[:layer, :]   # [layer, M] Tensor

            # Rank constraint: count of UE u on RBG m across previous layers < rank[u]
            u_idx = torch.arange(U).view(1, 1, -1)          # [1, 1, U]
            matches = (prev.unsqueeze(-1) == u_idx)          # [layer, M, U]
            counts  = matches.sum(dim=0)                     # [M, U]

            rank_t  = torch.tensor(self._rank, dtype=torch.float32)    # [U]
            rank_ok = counts < rank_t.unsqueeze(0)                     # [M, U]

            # Continuity constraint: if UE ever seen on RBG m, must be in last layer
            ever_seen     = counts > 0                                  # [M, U]
            last_alloc    = self._alloc[layer - 1, :]                  # [M]
            in_prev_layer = (last_alloc.unsqueeze(-1) ==               # [M, U]
                             torch.arange(U).unsqueeze(0))
            continuity_ok = (~ever_seen) | in_prev_layer               # [M, U]

            valid_ue = valid_ue & rank_ok & continuity_ok

        noop_col = torch.ones((M, 1), dtype=torch.bool)
        return torch.cat([valid_ue, noop_col], dim=1)  # [M, act_dim]

    # ── Internal: Reward (mirrors toy5g new_reward_compute) ──────────────────

    def _new_reward_compute(
        self, layer: int, masks: torch.Tensor
    ):
        """
        Returns (rewards_m [n_rbg], set_tp_per_rbg [n_rbg]) as Tensors.
        Mirrors toy5g DeterministicToy5GEnvAdapter.new_reward_compute exactly.
        """
        M    = self.n_rbg
        noop = self.noop
        U    = self.max_sched_ue

        rewards_m     = torch.zeros(M)
        set_tp_per_rbg = torch.zeros(M)

        for m in range(M):
            # Previous allocation on this RBG (layers 0..layer-1)
            prev_alloc: List[int] = []
            if layer > 0:
                for l_prev in range(layer):
                    u = int(self._alloc[l_prev, m].item())
                    if u != noop:
                        prev_alloc.append(u)

            T_prev = self._compute_set_tput(prev_alloc, m)

            chosen = int(self._alloc[layer, m].item())
            if chosen == noop:
                set_tp_per_rbg[m] = T_prev

            # Marginal gain for every valid UE
            raw_all = torch.zeros(U)
            for u in range(U):
                if not bool(masks[m, u].item()):
                    continue
                if self._buffers[u] <= 0:
                    continue
                curr_alloc = prev_alloc + [u]
                T_cur = self._compute_set_tput(curr_alloc, m)
                if u == chosen:
                    set_tp_per_rbg[m] = T_cur
                raw_all[u] = (T_cur - T_prev) / max(float(self._avg_tp[u]), self.eps)

            max_raw = float(raw_all.max().item())

            if max_raw > 0.0:
                if chosen == noop:
                    rewards_m[m] = 0.0
                else:
                    rewards_m[m] = float(
                        torch.clamp(raw_all[chosen] / max_raw, -1.0, 1.0).item()
                    )
            elif max_raw < 0.0:
                rewards_m[m] = 1.0 if chosen == noop else -1.0
            else:
                rewards_m[m] = 0.0

        return rewards_m, set_tp_per_rbg

    def _compute_set_tput(self, alloc_set: List[int], m: int) -> float:
        """
        Expected throughput of a set of co-scheduled UEs on RBG m.
        Mirrors toy5g compute_set_tput (with penalty / len division).
        """
        if len(alloc_set) == 0:
            return 0.0

        # Max pairwise kappa
        max_kappa = 0.0
        for i in range(len(alloc_set) - 1):
            u = alloc_set[i]
            for j in range(i + 1, len(alloc_set)):
                v = alloc_set[j]
                if u < self.max_sched_ue and v < self.max_sched_ue:
                    max_kappa = max(max_kappa, float(self._kappa[u, v]))

        penalty = (1.0 - max_kappa) / max(len(alloc_set), 1)

        tput = 0.0
        for u in alloc_set:
            if self._buffers[u] > 0 and u < self.max_sched_ue:
                tput += self._rate_mbps(u, m)

        return tput * penalty

    def _rate_mbps(self, u_local: int, m: int) -> float:
        """Estimated throughput [Mbps] for local UE u on RBG m. Mirrors toy5g."""
        mcs = int(self._mcs_sb[u_local, m])
        tbs_bytes = tbs_38214_bytes(
            mcs, self.PRB_PER_RBG,
            n_symb=self.N_SYMB,
            overhead_re_per_prb=self.OVERHEAD,
        )
        tti_s = 1e-3   # 1 ms slot
        return (tbs_bytes * 8.0) / 1e6 / max(tti_s, 1e-9)

    # ── Private TCP helpers ───────────────────────────────────────────────────

    def _recv(self) -> dict:
        return _recv_json(self._client_sock, self._rxbuf)

    def _send(self, obj: dict) -> None:
        _send_json(self._client_sock, obj)
