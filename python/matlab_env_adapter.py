"""
MatlabEnvAdapter
================
Drop-in replacement for DeterministicToy5GEnvAdapter.
Wraps TCP communication với MATLAB (nrDRLScheduler.m ở TrainingMode).

Interface is the same with toy5g for no change train_matlab.py despite of create env.

Protocol (layer-by-layer):
  MATLAB → Python : TTI_START  {tti}
  for l in 0..n_layers-1:
    MATLAB → Python : LAYER_OBS  {layer, obs:[obs_dim], masks:[n_rbg][act_dim]}
    Python → MATLAB : LAYER_ACT  {actions:[n_rbg]}   (0-indexed local UE / noop)
    MATLAB → Python : LAYER_REWARD {rewards:[n_rbg], next_obs:[obs_dim], next_masks:[n_rbg][act_dim]}
  MATLAB → Python : TTI_DONE  {metrics:{avg_cell_tput, jain, pf_utility,
                                         avg_layers_per_rbg, no_schedule_rate}}

obs_dim  = (5 + 2*n_rbg) * max_sched_ue   (matches toy5g _build_obs)
act_dim  = max_sched_ue + 1               (matches toy5g act_dim)
noop     = max_sched_ue                    (matches toy5g noop)
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional

import torch


# Reuse LayerContext dataclass definition (identical to toy5g)
@dataclass
class LayerContext:
    layer: int
    obs: torch.Tensor        # [obs_dim]
    masks_rbg: torch.Tensor  # [n_rbg, act_dim] bool


# ─── TCP helpers ──────────────────────────────────────────────────────────────

_NL = b"\n"


def _recv_json(sock: socket.socket, rxbuf: bytearray) -> dict:
    """Block until one newline-delimited JSON message is available."""
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
    """Send a single JSON object with newline terminator."""
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)


# ─── Adapter ──────────────────────────────────────────────────────────────────

class MatlabEnvAdapter:
    """
    Wraps MATLAB simulation as an RL environment with the same interface
    as DeterministicToy5GEnvAdapter.

    Key attributes (mirrors toy5g):
        max_sched_ue : int   – eligible UEs per TTI (= MATLAB MaxUEs, default 16)
        n_ue         : int   – same as max_sched_ue (for eval compatibility)
        n_layers     : int   – MU-MIMO spatial layers (= MATLAB NumLayers)
        n_rbg        : int   – Resource Block Groups
        obs_dim      : int   – (5 + 2*n_rbg) * max_sched_ue
        act_dim      : int   – max_sched_ue + 1
        noop         : int   – max_sched_ue  (no-allocation action)
        eps          : float – small constant
        _alloc       : Tensor[n_layers, n_rbg]  – current TTI allocation
        _last_metrics: dict  – metrics received from MATLAB after finish_tti()
    """

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
        self.n_ue = max_sched_ue          # alias for eval compatibility
        self.n_layers = n_layers
        self.n_rbg = n_rbg
        self.noop = max_sched_ue          # same as toy5g
        self.act_dim = max_sched_ue + 1   # same as toy5g
        self.obs_dim = (5 + 2 * n_rbg) * max_sched_ue
        self.eps = float(eps)
        self.verbose = verbose

        # Internal state (mirrors toy5g)
        self._alloc = torch.full(
            (n_layers, n_rbg), self.noop, dtype=torch.long
        )
        self._last_transitions: List[Dict] = []
        self._cur_layer: Optional[int] = None
        self._last_metrics: dict = {}

        # TCP server: wait for MATLAB to connect
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

    # ── Public API (identical to toy5g) ────────────────────────────────────

    def reset(self):
        """Reset internal state (call once before the training loop)."""
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None
        self._last_metrics = {}

    def begin_tti(self):
        """
        Wait for TTI_START from MATLAB.
        Resets per-TTI allocation cache.
        """
        self._alloc.fill_(self.noop)
        self._last_transitions = []
        self._cur_layer = None

        msg = self._recv()
        if msg.get("type") != "TTI_START":
            got = msg.get("type")
            hint = ""
            if got == "TTI_OBS":
                hint = (
                    " MATLAB is using the old protocol. "
                    "Set drlScheduler.TrainingMode = true; in MU_MIMO.m before connecting."
                )
            raise RuntimeError(
                f"[MatlabEnv] Expected TTI_START, got: {got}.{hint}"
            )
        # Validate n_layers / n_rbg if MATLAB sends them (first TTI)
        if "n_layers" in msg and msg["n_layers"] != self.n_layers:
            raise RuntimeError(
                f"[MatlabEnv] n_layers mismatch: MATLAB={msg['n_layers']}, "
                f"Python={self.n_layers}. Pass --n_layers {msg['n_layers']} to train_matlab.py."
            )
        if "n_rbg" in msg and msg["n_rbg"] != self.n_rbg:
            raise RuntimeError(
                f"[MatlabEnv] n_rbg mismatch: MATLAB={msg['n_rbg']}, "
                f"Python={self.n_rbg}. Pass --n_rbg {msg['n_rbg']} to train_matlab.py."
            )
        if self.verbose:
            tti = msg.get("tti", "?")
            print(f"[MatlabEnv] === TTI {tti} ===")

    def layer_iter(self) -> Iterator[LayerContext]:
        """
        Yield one LayerContext per spatial layer.
        Receives LAYER_OBS from MATLAB for each layer.
        """
        for l in range(self.n_layers):
            self._cur_layer = l

            msg = self._recv()
            if msg.get("type") != "LAYER_OBS":
                raise RuntimeError(
                    f"[MatlabEnv] Expected LAYER_OBS (layer {l}), "
                    f"got: {msg.get('type')}"
                )
            if msg.get("layer") != l:
                raise RuntimeError(
                    f"[MatlabEnv] Layer mismatch: expected {l}, "
                    f"got {msg.get('layer')}"
                )

            obs = torch.tensor(msg["obs"], dtype=torch.float32)    # [obs_dim]
            masks = torch.tensor(
                msg["masks"], dtype=torch.bool
            )  # [n_rbg, act_dim]

            # Validate shapes
            assert obs.shape == (self.obs_dim,), \
                f"obs shape {obs.shape} != ({self.obs_dim},)"
            assert masks.shape == (self.n_rbg, self.act_dim), \
                f"masks shape {masks.shape} != ({self.n_rbg}, {self.act_dim})"

            yield LayerContext(layer=l, obs=obs, masks_rbg=masks)

    def apply_layer_actions(
        self, layer_ctx: LayerContext, actions_rbg: torch.Tensor
    ):
        """
        Store actions in _alloc and send LAYER_ACT to MATLAB.

        actions_rbg : Tensor[n_rbg] int64 on CPU
                      values in [0..act_dim-1], noop = max_sched_ue
        """
        l = layer_ctx.layer
        a = actions_rbg.cpu().long().clamp(0, self.act_dim - 1)
        self._alloc[l, :] = a

        self._send({"type": "LAYER_ACT", "actions": a.tolist()})

    def compute_layer_transitions(
        self, layer_ctx: LayerContext
    ) -> List[Dict]:
        """
        Receive LAYER_REWARD from MATLAB, package into per-RBG transition dicts.
        Format matches toy5g _compute_layer_transitions output.
        """
        msg = self._recv()
        if msg.get("type") != "LAYER_REWARD":
            raise RuntimeError(
                f"[MatlabEnv] Expected LAYER_REWARD, got: {msg.get('type')}"
            )

        rewards = msg["rewards"]            # list[n_rbg] float
        next_obs = torch.tensor(
            msg["next_obs"], dtype=torch.float32
        )  # [obs_dim]
        next_masks = torch.tensor(
            msg["next_masks"], dtype=torch.bool
        )  # [n_rbg, act_dim]

        obs = layer_ctx.obs
        masks = layer_ctx.masks_rbg
        l = layer_ctx.layer

        out: List[Dict] = []
        for m in range(self.n_rbg):
            tr = {
                "observation":      obs.cpu(),
                "next_observation": next_obs.cpu(),
                "rbg_index":        torch.tensor(m, dtype=torch.long),
                "action":           torch.tensor(
                                        int(self._alloc[l, m].item()),
                                        dtype=torch.long
                                    ),
                "reward":           torch.tensor(
                                        float(rewards[m]), dtype=torch.float32
                                    ),
                "action_mask":      masks[m].cpu(),           # [act_dim] bool
                "next_action_mask": next_masks[m].cpu(),      # [act_dim] bool
            }
            out.append(tr)
            self._last_transitions.append(tr)
        return out

    def finish_tti(self):
        """
        Wait for TTI_DONE from MATLAB.
        Stores evaluation metrics sent by MATLAB.
        """
        msg = self._recv()
        if msg.get("type") != "TTI_DONE":
            raise RuntimeError(
                f"[MatlabEnv] Expected TTI_DONE, got: {msg.get('type')}"
            )
        self._last_metrics = msg.get("metrics", {})

    # ── Private helpers ────────────────────────────────────────────────────

    def _recv(self) -> dict:
        return _recv_json(self._client_sock, self._rxbuf)

    def _send(self, obj: dict) -> None:
        _send_json(self._client_sock, obj)

    def close(self):
        """Close TCP connection."""
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
