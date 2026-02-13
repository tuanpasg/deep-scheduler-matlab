"""
DRL Training Server for MATLAB 5G MU-MIMO Scheduler
Uses TTI_OBS protocol: MATLAB sends one observation per TTI, Python generates full allocation matrix
"""

import argparse
import socket
import json
import math
import numpy as np
import torch
import os
from collections import deque
from datetime import datetime

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    ensure_nonempty_mask,
    apply_action_mask_to_logits,
)
from train_2_logging import (
    init_eval_log,
    init_train_log,
    append_eval,
    plot_eval,
    plot_training,
    save_logs,
)

CQI_TO_SE = [
    0.0000, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766,
    1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547
]

# 38.214-inspired MCS Table (Qm, R*1024)
MCS_TABLE = [
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379),
    (4, 449), (4, 526), (4, 602), (4, 679), (6, 340), (6, 378),
    (6, 434), (6, 490), (6, 553), (6, 616), (6, 658), (8, 438),
    (8, 466), (8, 517), (8, 567), (8, 616), (8, 666), (8, 719),
    (8, 772), (8, 822), (8, 873), (8, 910), (8, 948),
]

# CQI to MCS mapping (approximate)
CQI_TO_MCS = [0, 0, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28]


def tbs_38214_bytes(mcs_idx, n_prb, n_symb=14, n_layers=1, overhead_re_per_prb=18):
    """Compute TBS in bytes using 38.214 approximation."""
    if n_prb <= 0 or mcs_idx < 0:
        return 0
    mcs_idx = int(max(0, min(28, mcs_idx)))
    Qm, R1024 = MCS_TABLE[mcs_idx]
    R = R1024 / 1024.0
    N_re_per_prb = max(0, 12 * n_symb - overhead_re_per_prb)
    Ninfo = R * Qm * N_re_per_prb * int(n_prb) * n_layers
    if Ninfo <= 0:
        return 0
    if Ninfo <= 3824:
        TBS_bits = int(6 * math.ceil(Ninfo / 6.0))
    else:
        TBS_bits = int(6 * math.ceil((Ninfo - 24) / 6.0))
    TBS_bits = max(TBS_bits, 24)
    return TBS_bits // 8


class ReplayBuffer:
    """Experience replay buffer for DRL training.
    
    Stores per-RBG experiences for DSACD paper-style training.
    Each entry: (state, rbg_idx, action, reward, next_state, mask, next_mask, done)
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask=None, next_mask=None):
        """Store experience tuple. Ensure all tensors are on CPU.
        
        For DSACD, we expand layer-step experiences into per-RBG experiences.
        action: [n_rbg] actions
        mask: [n_rbg, n_actions] action mask (optional)
        """
        # Move tensors to CPU before storing to avoid device mismatch
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(action, torch.Tensor):
            action = action.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu()
        if isinstance(next_mask, torch.Tensor):
            next_mask = next_mask.cpu()
            
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size):
        """Sample random batch from buffer and expand to per-RBG format for DSACD."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        
        # Masks (may be None for old buffer entries)
        masks = [b[5] for b in batch]
        next_masks = [b[6] for b in batch]
        
        return states, actions, rewards, next_states, dones, masks, next_masks
    
    def __len__(self):
        return len(self.buffer)


class MATLABDRLTrainer:
    """DRL Training server that communicates with MATLAB via TCP."""
    
    def __init__(self, port=5555, device='cuda', verbose=True):
        self.port = port
        self.device = device
        self.verbose = verbose
        
        # Network setup
        self.actor = None
        self.updater = None
        self.hyperparams = None
        
        # Training state
        self.replay_buffer = None
        self.total_steps = 0  # TTI count
        self.layer_steps = 0  # Layer-step count (for training freq/epsilon)
        self.episode_count = 0
        self.training_enabled = False
        
        # Communication
        self.server_socket = None
        self.client_socket = None
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Allocation tracking for reward alignment
        self.prev_allocation_matrix = None
        self.last_allocation_matrix = None
        
        # Fixed network dimensions (set on first initialization)
        self.fixed_n_actions = None  # Will be set to max_ues + 1 on first init
        
        # Logging
        self.train_log = None
        self.eval_log = None
        
    def initialize_networks(self, obs_dim, n_rbg, n_actions_per_rbg, hidden_dim=256, 
                           n_quantiles=32, gamma=0.99, tau=0.005, lr_actor=3e-4, 
                           lr_critic=3e-4, lr_alpha=3e-4, beta=0.98, fallback_action=-1):
        """Initialize actor, critic, and updater."""
        hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
        max_paired_ues = hp.get('max_paired_ues', 4)
        
        # Store fixed action dimension for the lifetime of the trainer
        self.fixed_n_actions = n_actions_per_rbg
        
        print(f"[Trainer] Initializing networks...")
        print(f"  - Observation dim: {obs_dim}")
        print(f"  - Number of RBGs: {n_rbg}")
        print(f"  - Actions per RBG: {n_actions_per_rbg}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - N quantiles: {n_quantiles}")
        print(f"  - Max paired UEs per RBG: {max_paired_ues}")
        
        # Create networks
        self.actor = MultiBranchActor(
            obs_dim=obs_dim,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            hidden=hidden_dim
        ).to(self.device)
        
        # Q-networks (q1, q2) and targets
        q1 = MultiBranchQuantileCritic(
            obs_dim=obs_dim,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            n_quantiles=n_quantiles,
            hidden=hidden_dim
        ).to(self.device)
        
        q2 = MultiBranchQuantileCritic(
            obs_dim=obs_dim,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            n_quantiles=n_quantiles,
            hidden=hidden_dim
        ).to(self.device)
        
        q1_target = MultiBranchQuantileCritic(
            obs_dim=obs_dim,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            n_quantiles=n_quantiles,
            hidden=hidden_dim
        ).to(self.device)
        
        q2_target = MultiBranchQuantileCritic(
            obs_dim=obs_dim,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            n_quantiles=n_quantiles,
            hidden=hidden_dim
        ).to(self.device)
        
        # Copy weights to targets
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        
        # Hyperparameters
        hp = DSACDHyperParams(
            n_quantiles=n_quantiles,
            beta=beta,
            gamma=gamma,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            fallback_action=fallback_action,
        )
        
        # Updater
        self.updater = DSACDUpdater(
            actor=self.actor,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            n_rbg=n_rbg,
            act_dim=n_actions_per_rbg,
            hp=hp,
            device=str(self.device)
        )
        
        # Replay buffer (do NOT override if already created)
        if self.replay_buffer is None:
            hp_dict = self.hyperparams if isinstance(self.hyperparams, dict) else {}
            cap = int(hp_dict.get("rb_capacity", 100000))
            self.replay_buffer = ReplayBuffer(capacity=cap)

        # Initialize logging
        self.train_log = init_train_log()
        self.eval_log = init_eval_log()

        self.training_enabled = True
        
        # Log model parameters
        self._log_model_parameters()
        
        print(f"[Trainer] Networks initialized successfully")

        # Apply deferred checkpoint weights (if any)
        self._apply_pending_checkpoint_if_any()
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Safe checkpoint loader:
        - If networks are NOT initialized yet, only load counters now and defer weights.
        - If networks ARE initialized, load weights immediately.
        """
        if not checkpoint_path:
            return False
        if not os.path.exists(checkpoint_path):
            print(f"[Trainer] Warning: Checkpoint not found: {checkpoint_path}")
            return False

        # Always load meta/counters early
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.layer_steps = int(ckpt.get("layer_steps", self.total_steps * 16))
        self.episode_count = int(ckpt.get("episode_count", 0))
        
        # Restore fixed_n_actions if saved (critical for mask consistency)
        if "fixed_n_actions" in ckpt and ckpt["fixed_n_actions"] is not None:
            self.fixed_n_actions = int(ckpt["fixed_n_actions"])
            print(f"[Trainer] Restored fixed_n_actions = {self.fixed_n_actions}")

        # Defer or apply weights
        if self.actor is None or self.updater is None:
            # Networks not ready -> defer
            setattr(self, "_pending_ckpt_path", checkpoint_path)
            print(f"[Trainer] Checkpoint metadata loaded, weights deferred until networks init.")
            print(f"[Trainer] Pending checkpoint: {checkpoint_path}")
            print(f"[Trainer] Resumed counters: TTI={self.total_steps}, LayerStep={self.layer_steps}")
            return True

        # Networks ready -> load now
        if "actor" in ckpt and ckpt["actor"]:
            self.actor.load_state_dict(ckpt["actor"])
        if "updater" in ckpt and ckpt["updater"]:
            updater_state = ckpt["updater"]
            if 'q1' in updater_state:
                self.updater.q1.load_state_dict(updater_state['q1'])
            if 'q2' in updater_state:
                self.updater.q2.load_state_dict(updater_state['q2'])
            if 'q1_target' in updater_state:
                self.updater.q1_t.load_state_dict(updater_state['q1_target'])
            if 'q2_target' in updater_state:
                self.updater.q2_t.load_state_dict(updater_state['q2_target'])
            if 'log_alpha' in updater_state:
                self.updater.log_alpha.data.copy_(updater_state['log_alpha'])
            if 'opt_actor' in updater_state:
                self.updater.opt_actor.load_state_dict(updater_state['opt_actor'])
            if 'opt_critic' in updater_state:
                self.updater.opt_critic.load_state_dict(updater_state['opt_critic'])
            if 'opt_alpha' in updater_state:
                self.updater.opt_alpha.load_state_dict(updater_state['opt_alpha'])

        print(f"[Trainer] Loaded checkpoint (weights + counters) from {checkpoint_path}")
        print(f"[Trainer] Resumed: TTI={self.total_steps}, LayerStep={self.layer_steps}")
        return True
    

    def _apply_pending_checkpoint_if_any(self):
        """Helper: apply deferred checkpoint once actor/updater exist."""
        pending = getattr(self, "_pending_ckpt_path", None)
        if not pending:
            return
        if self.actor is None or self.updater is None:
            return
        if not os.path.exists(pending):
            print(f"[Trainer] Warning: pending checkpoint path missing: {pending}")
            setattr(self, "_pending_ckpt_path", None)
            return

        ckpt = torch.load(pending, map_location=self.device)
        if "actor" in ckpt and ckpt["actor"]:
            self.actor.load_state_dict(ckpt["actor"])
        if "updater" in ckpt and ckpt["updater"]:
            updater_state = ckpt["updater"]
            if 'q1' in updater_state:
                self.updater.q1.load_state_dict(updater_state['q1'])
            if 'q2' in updater_state:
                self.updater.q2.load_state_dict(updater_state['q2'])
            if 'q1_target' in updater_state:
                self.updater.q1_t.load_state_dict(updater_state['q1_target'])
            if 'q2_target' in updater_state:
                self.updater.q2_t.load_state_dict(updater_state['q2_target'])
            if 'log_alpha' in updater_state:
                self.updater.log_alpha.data.copy_(updater_state['log_alpha'])
            if 'opt_actor' in updater_state:
                self.updater.opt_actor.load_state_dict(updater_state['opt_actor'])
            if 'opt_critic' in updater_state:
                self.updater.opt_critic.load_state_dict(updater_state['opt_critic'])
            if 'opt_alpha' in updater_state:
                self.updater.opt_alpha.load_state_dict(updater_state['opt_alpha'])

        print(f"[Trainer] Applied deferred checkpoint weights from {pending}")
        setattr(self, "_pending_ckpt_path", None)
    
    def _log_model_parameters(self):
        """Log the number of parameters in each model component."""
        def count_params(model):
            """Count total and trainable parameters."""
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable
        
        def format_params(num):
            """Format parameter count with appropriate suffix (K, M, B)."""
            if num >= 1e9:
                return f"{num / 1e9:.2f}B"
            elif num >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif num >= 1e3:
                return f"{num / 1e3:.2f}K"
            else:
                return str(num)
        
        print("\n" + "=" * 60)
        print("MODEL PARAMETERS SUMMARY")
        print("=" * 60)
        
        total_all = 0
        trainable_all = 0
        
        # Actor
        if self.actor is not None:
            total, trainable = count_params(self.actor)
            total_all += total
            trainable_all += trainable
            print(f"  Actor:      {format_params(total):>10} total, {format_params(trainable):>10} trainable")
        
        # Critics (from updater)
        if self.updater is not None:
            # Q1
            total, trainable = count_params(self.updater.q1)
            total_all += total
            trainable_all += trainable
            print(f"  Critic Q1:  {format_params(total):>10} total, {format_params(trainable):>10} trainable")
            
            # Q2
            total, trainable = count_params(self.updater.q2)
            total_all += total
            trainable_all += trainable
            print(f"  Critic Q2:  {format_params(total):>10} total, {format_params(trainable):>10} trainable")
            
            # Q1 Target (not trainable but counts in model size)
            total_t1, _ = count_params(self.updater.q1_t)
            print(f"  Q1 Target:  {format_params(total_t1):>10} total (frozen)")
            
            # Q2 Target
            total_t2, _ = count_params(self.updater.q2_t)
            print(f"  Q2 Target:  {format_params(total_t2):>10} total (frozen)")
        
        print("-" * 60)
        print(f"  TOTAL (trainable networks): {format_params(total_all):>10} ({total_all:,} params)")
        print(f"  TRAINABLE:                  {format_params(trainable_all):>10} ({trainable_all:,} params)")
        print("=" * 60 + "\n")
    
    def save_checkpoint(self, save_dir='checkpoints'):
        """Save current model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(save_dir, f"checkpoint_layer_{self.layer_steps}_{timestamp}.pt")
        
        # DSACDUpdater doesn't have state_dict, save components separately
        updater_state = {}
        if self.updater is not None:
            updater_state = {
                'q1': self.updater.q1.state_dict(),
                'q2': self.updater.q2.state_dict(),
                'q1_target': self.updater.q1_t.state_dict(),
                'q2_target': self.updater.q2_t.state_dict(),
                'log_alpha': self.updater.log_alpha.detach().cpu(),
                'opt_actor': self.updater.opt_actor.state_dict(),
                'opt_critic': self.updater.opt_critic.state_dict(),
                'opt_alpha': self.updater.opt_alpha.state_dict(),
            }
        
        torch.save({
            'actor': self.actor.state_dict() if self.actor else {},
            'updater': updater_state,
            'total_steps': self.total_steps,
            'layer_steps': self.layer_steps,
            'episode_count': self.episode_count,
            'fixed_n_actions': self.fixed_n_actions,  # Critical: save network action dim
        }, checkpoint_path)
        
        print(f"[Trainer] Saved checkpoint to {checkpoint_path}")
        
        # Also save logs
        self.save_logs(save_dir)
        
        return checkpoint_path
    
    def save_logs(self, save_dir='checkpoints'):
        """Save training and evaluation logs to JSON files."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save eval_log
        if self.eval_log is not None:
            eval_log_path = os.path.join(save_dir, f"eval_log_tti{self.total_steps}_{timestamp}.json")
            try:
                with open(eval_log_path, 'w') as f:
                    json.dump(self.eval_log, f, indent=2)
                print(f"[Trainer] Saved eval log to {eval_log_path}")
            except Exception as e:
                print(f"[Trainer] Warning: Could not save eval log: {e}")
        
        # Save train_log
        if self.train_log is not None:
            train_log_path = os.path.join(save_dir, f"train_log_tti{self.total_steps}_{timestamp}.json")
            try:
                with open(train_log_path, 'w') as f:
                    json.dump(self.train_log, f, indent=2)
                print(f"[Trainer] Saved train log to {train_log_path}")
            except Exception as e:
                print(f"[Trainer] Warning: Could not save train log: {e}")
    
    def build_observation(self, data, max_ues, num_rbg, allocated_rbg=None, cross_corr=None, 
                          prev_alloc=None, layer=0):
        """
        Build observation EXACTLY matching _build_obs from toy5g_env_adapter.py:
        
        Per-UE features (5 scalars):
        1. norm_past_avg_tp = avg_tp / max_rate (max_rate = 1100 Mbps fixed)
        2. norm_ue_rank = rank / max_ue_rank (max_ue_rank = 2)
        3. norm_allocated_rbgs = alloc_count / n_rbg (computed per layer)
        4. norm_buffer = buf / max(buf) (dynamic max across UEs)
        5. norm_wb_mcs = mcs / 28
        
        Per-UE-per-RBG features:
        6. norm_subband_mcs [U, n_rbg] = subband_mcs / 28
        7. max_corr_feat [U, n_rbg] = max correlation with scheduled UEs on each RBG
        
        Total obs_dim = (5 + 2*n_rbg) * max_ues
        
        Args:
            data: dict with fields from MATLAB or list (old format)
            max_ues: Number of UEs
            num_rbg: Number of RBGs
            allocated_rbg: [max_ues][num_rbg] allocation tracking (1 if allocated)
            cross_corr: [max_ues x max_ues x num_rbg] correlation matrix from MATLAB
            prev_alloc: list of [num_rbg] per layer - UE indices (0-based) or max_ues (NOOP)
            layer: Current layer index
        """
        eps = 1e-6
        MAX_RATE = 1100.0  # Mbps - fixed max rate (matches toy5g_env_adapter)
        MAX_UE_RANK = 2.0  # Max rank
        MAX_MCS = 28.0     # Max MCS index
        
        # Support both old format (list) and new format (dict)
        if isinstance(data, dict):
            # New format with separate fields from MATLAB
            avg_throughput = np.array(data.get('avg_throughput', [0] * max_ues), dtype=np.float32).flatten()
            rank_ue = np.array(data.get('rank', [1] * max_ues), dtype=np.float32).flatten()
            buffer_status = np.array(data.get('buffer', [0] * max_ues), dtype=np.float32).flatten()
            
            # MATLAB sends CQI (0-15), convert to MCS (0-28)
            wideband_cqi = np.array(data.get('wideband_cqi', [7] * max_ues), dtype=np.float32).flatten()
            wideband_mcs = wideband_cqi * (28.0 / 15.0)  # CQI -> MCS approximation
            
            subband_cqi = np.array(data.get('subband_cqi', np.zeros((max_ues, num_rbg))), dtype=np.float32)
            if subband_cqi.ndim == 1:
                subband_cqi = np.tile(wideband_cqi[:, np.newaxis], (1, num_rbg))
            subband_mcs = subband_cqi * (28.0 / 15.0)  # CQI -> MCS
            
            if cross_corr is None:
                cross_corr = np.array(data.get('cross_corr', np.zeros((max_ues, max_ues, num_rbg))), dtype=np.float32)
        else:
            # Old format: features is [max_ues, feat_dim] list
            features = data
            x = np.array(features, dtype=np.float32)
            feat_dim = x.shape[1] if x.ndim == 2 else 0
            
            avg_throughput = x[:, 0] if feat_dim > 0 else np.zeros(max_ues)
            rank_ue = x[:, 1] if feat_dim > 1 else np.ones(max_ues)
            buffer_status = x[:, 3] if feat_dim > 3 else np.zeros(max_ues)
            wideband_mcs = x[:, 4] * (28.0 / 15.0) if feat_dim > 4 else np.zeros(max_ues)
            
            # Subband MCS from columns 5+
            if feat_dim > 5:
                subband_cqi = x[:, 5:5 + num_rbg]
                if subband_cqi.shape[1] < num_rbg:
                    pad = np.tile(wideband_mcs[:, np.newaxis] * (15.0/28.0), (1, num_rbg - subband_cqi.shape[1]))
                    subband_cqi = np.concatenate([subband_cqi, pad], axis=1)
            else:
                subband_cqi = np.tile(wideband_mcs[:, np.newaxis] * (15.0/28.0), (1, num_rbg))
            subband_mcs = subband_cqi * (28.0 / 15.0)
            
            if cross_corr is None:
                cross_corr = np.zeros((max_ues, max_ues, num_rbg), dtype=np.float32)
        
        # Ensure arrays are correct size
        avg_throughput = avg_throughput[:max_ues] if len(avg_throughput) >= max_ues else np.pad(avg_throughput, (0, max_ues - len(avg_throughput)))
        rank_ue = rank_ue[:max_ues] if len(rank_ue) >= max_ues else np.pad(rank_ue, (0, max_ues - len(rank_ue)), constant_values=1)
        buffer_status = buffer_status[:max_ues] if len(buffer_status) >= max_ues else np.pad(buffer_status, (0, max_ues - len(buffer_status)))
        wideband_mcs = wideband_mcs[:max_ues] if len(wideband_mcs) >= max_ues else np.pad(wideband_mcs, (0, max_ues - len(wideband_mcs)))
        subband_mcs = subband_mcs[:max_ues, :num_rbg]
        if subband_mcs.shape[0] < max_ues:
            pad = np.zeros((max_ues - subband_mcs.shape[0], num_rbg))
            subband_mcs = np.concatenate([subband_mcs, pad], axis=0)
        if subband_mcs.shape[1] < num_rbg:
            pad = np.zeros((max_ues, num_rbg - subband_mcs.shape[1]))
            subband_mcs = np.concatenate([subband_mcs, pad], axis=1)
        
        # =====================================================
        # 1. norm_past_avg_tp = avg_tp / max_rate (FIXED max_rate = 1100 Mbps)
        # =====================================================
        norm_past_avg_tp = np.clip(avg_throughput / MAX_RATE, 0, 1)
        
        # =====================================================
        # 2. norm_ue_rank = rank / max_ue_rank
        # =====================================================
        norm_ue_rank = np.clip(rank_ue / MAX_UE_RANK, 0, 1)
        
        # =====================================================
        # 3. norm_allocated_rbgs = alloc_count / n_rbg (per layer)
        # =====================================================
        if layer > 0 and prev_alloc is not None:
            # Count how many times each UE was allocated in previous layers
            alloc_counts = np.zeros(max_ues, dtype=np.float32)
            for l in range(layer):
                if l < len(prev_alloc):
                    for rbg in range(num_rbg):
                        ue_idx = prev_alloc[l][rbg] if isinstance(prev_alloc[l], list) else int(prev_alloc[l, rbg])
                        if ue_idx < max_ues:  # Valid UE (not NOOP)
                            alloc_counts[ue_idx] += 1
            norm_allocated_rbgs = alloc_counts / max(num_rbg, 1)
        elif allocated_rbg is not None:
            # Fallback: use allocated_rbg matrix
            alloc_counts = np.array([sum(allocated_rbg[u]) for u in range(max_ues)], dtype=np.float32)
            norm_allocated_rbgs = alloc_counts / max(num_rbg, 1)
        else:
            norm_allocated_rbgs = np.zeros(max_ues, dtype=np.float32)
        
        # =====================================================
        # 4. norm_buffer = buf / max(buf) (DYNAMIC max)
        # =====================================================
        max_buf = max(np.max(buffer_status), eps)
        norm_buffer = np.clip(buffer_status / max_buf, 0, 1)
        
        # =====================================================
        # 5. norm_wb_mcs = mcs / 28
        # =====================================================
        norm_wb_mcs = np.clip(wideband_mcs / MAX_MCS, 0, 1)
        
        # =====================================================
        # 6. norm_subband_mcs [U, n_rbg] = subband_mcs / 28
        # =====================================================
        norm_subband_mcs = np.clip(subband_mcs / MAX_MCS, 0, 1)
        
        # =====================================================
        # 7. max_corr_feat [U, n_rbg] = max correlation with scheduled UEs
        # =====================================================
        max_corr_feat = np.zeros((max_ues, num_rbg), dtype=np.float32)
        
        if layer > 0 and prev_alloc is not None and cross_corr is not None and cross_corr.size > 0:
            for m in range(num_rbg):
                # Get UEs scheduled on this RBG in previous layers
                scheduled_ues = []
                for l in range(layer):
                    if l < len(prev_alloc):
                        ue_idx = prev_alloc[l][m] if isinstance(prev_alloc[l], list) else int(prev_alloc[l, m])
                        if ue_idx < max_ues:  # Valid UE (not NOOP)
                            scheduled_ues.append(ue_idx)
                
                if scheduled_ues:
                    # For each candidate UE, find max correlation with any scheduled UE
                    for u in range(max_ues):
                        max_corr = 0.0
                        for sched_u in scheduled_ues:
                            if (u < cross_corr.shape[0] and sched_u < cross_corr.shape[1] 
                                and m < cross_corr.shape[2]):
                                corr = float(cross_corr[u, sched_u, m])
                                max_corr = max(max_corr, corr)
                        max_corr_feat[u, m] = max_corr
        
        # =====================================================
        # Stack: [U, 5] -> concat -> [U, 5 + 2*n_rbg] -> flatten
        # =====================================================
        ue_scalar_feats = np.stack([
            norm_past_avg_tp,      # [U]
            norm_ue_rank,          # [U]
            norm_allocated_rbgs,   # [U]
            norm_buffer,           # [U]
            norm_wb_mcs            # [U]
        ], axis=1)  # [U, 5]
        
        # Concatenate: [U, 5] + [U, n_rbg] + [U, n_rbg] = [U, 5 + 2*n_rbg]
        ue_feats = np.concatenate([ue_scalar_feats, norm_subband_mcs, max_corr_feat], axis=1)
        
        # Flatten to 1D: obs_dim = (5 + 2*n_rbg) * max_ues
        obs = ue_feats.reshape(-1).astype(np.float32)
        
        return torch.tensor(obs, dtype=torch.float32)
    
    def build_action_mask(self,
                          num_rbg, max_ues, eligible_ues, buffer_status,
                          ue_set_per_rbg, rank_used_per_rbg_ue,
                          max_ue_layers_per_rbg=8, max_rank_per_ue_per_rbg=2,
                          last_layer_per_rbg_ue=None, current_layer_idx=0,
                          layers_used_per_rbg=None,
                          features=None, num_subbands=1, num_rbs=None,
                          rbg_size=None, subband_size=None,
                          mu_corr_threshold=0.9, min_cqi=1):
        """
        Mask per RBG with Table 4 + CONTINUOUS LAYER + PACK FROM LAYER 0 constraints:
          - max 8 distinct UEs per RBG
          - max rank 2 per UE per RBG
          - NO_ALLOC always valid
          - CONTINUOUS LAYER: UE can only be scheduled on layer L if:
            * It's new to this RBG, OR
            * It was scheduled on layer L-1 (immediately preceding)
          - PACK FROM LAYER 0: New UEs can only start at next available layer
            * If layers_used=3, new UE can only start at layer 3
            * This prevents gaps like: layer 0=UE1, layer 5=UE2 (bad!)
            → Standard-compliant PDSCH layer mapping
        """
        # Use fixed network dimension if available, otherwise fall back to max_ues + 1
        if self.fixed_n_actions is not None:
            n_actions = self.fixed_n_actions
        else:
            n_actions = max_ues + 1
        NO_ALLOC = n_actions - 1  # Last action is NO_ALLOC

        mask = torch.zeros(num_rbg, n_actions, dtype=torch.bool)

        eligible_set = set(eligible_ues)

        if num_rbs is None:
            num_rbs = num_rbg
        if rbg_size is None:
            rbg_size = max(1, int(math.ceil(num_rbs / max(num_rbg, 1))))
        if subband_size is None:
            subband_size = max(1, int(math.ceil(num_rbs / max(num_subbands, 1))))
        min_se_norm = CQI_TO_SE[min_cqi] / CQI_TO_SE[-1] if min_cqi < len(CQI_TO_SE) else 0.0

        for rbg in range(num_rbg):
            # NO_ALLOC always valid
            mask[rbg, NO_ALLOC] = True

            distinct_cnt = len(ue_set_per_rbg[rbg])
            
            # Get current layers used on this RBG
            rbg_layers_used = layers_used_per_rbg[rbg] if layers_used_per_rbg else 0

            for ue_id in eligible_ues:
                if ue_id <= 0 or ue_id > max_ues:
                    continue
                ue0 = ue_id - 1  # 0-based action index

                # must have buffer
                if buffer_status[ue0] <= 0:
                    continue

                # Subband-aware gating (paper-style robustness)
                if features is not None and num_subbands > 0:
                    rb_start = rbg * rbg_size
                    sb_idx = int(rb_start / subband_size)
                    if sb_idx >= num_subbands:
                        sb_idx = num_subbands - 1
                    feat_row = features[ue0] if ue0 < len(features) else None
                    if feat_row is not None:
                        idx_se = 5 + sb_idx
                        idx_rho = 5 + num_subbands + sb_idx
                        if idx_se < len(feat_row):
                            se_norm = feat_row[idx_se]
                            if se_norm > 0 and se_norm < min_se_norm:
                                continue
                        if idx_rho < len(feat_row):
                            rho_val = feat_row[idx_rho]
                            if rho_val > 0 and rho_val > mu_corr_threshold:
                                continue

                if ue0 in ue_set_per_rbg[rbg]:
                    # UE already on this RBG - enforce CONTINUOUS LAYER constraint
                    # Only allow if current layer immediately follows last assigned layer
                    if last_layer_per_rbg_ue is not None:
                        last_layer = last_layer_per_rbg_ue[rbg][ue0]
                        if current_layer_idx != last_layer + 1:
                            # Non-continuous! Skip to ensure single MAC PDU per UE
                            continue
                    
                    # Also check rank constraint (max 2 layers per UE per RBG)
                    if rank_used_per_rbg_ue[rbg][ue0] < max_rank_per_ue_per_rbg:
                        mask[rbg, ue0] = True
                else:
                    # NEW UE: Must start at next available layer (pack from layer 0)
                    # Only allow if current_layer_idx == layers_used (no gaps!)
                    if current_layer_idx != rbg_layers_used:
                        # Would create a gap - skip this UE
                        continue
                    
                    # Also check distinct UE count < 8
                    if distinct_cnt < max_ue_layers_per_rbg:
                        mask[rbg, ue0] = True

            # Fallback: if only NO_ALLOC is valid, relax subband gating for this RBG
            if mask[rbg].sum().item() == 1 and mask[rbg, NO_ALLOC]:
                for ue_id in eligible_ues:
                    if ue_id <= 0 or ue_id > max_ues:
                        continue
                    ue0 = ue_id - 1
                    if buffer_status[ue0] <= 0:
                        continue
                    if ue0 in ue_set_per_rbg[rbg]:
                        if last_layer_per_rbg_ue is not None:
                            last_layer = last_layer_per_rbg_ue[rbg][ue0]
                            if current_layer_idx != last_layer + 1:
                                continue
                        if rank_used_per_rbg_ue[rbg][ue0] < max_rank_per_ue_per_rbg:
                            mask[rbg, ue0] = True
                    else:
                        if current_layer_idx != rbg_layers_used:
                            continue
                        if distinct_cnt < max_ue_layers_per_rbg:
                            mask[rbg, ue0] = True

        mask = ensure_nonempty_mask(mask)
        return mask
    
    def select_action(self, obs, mask, explore=True, epsilon=0.1, layer_idx=0):
        """
        Select action using actor network with exploration.
        
        Args:
            obs: Observation tensor
            mask: Action mask [n_rbg, n_actions]
            explore: Whether to add exploration noise
            epsilon: Epsilon for epsilon-greedy exploration
            layer_idx: Current layer index (for logging)
            
        Returns:
            actions: Selected actions [n_rbg]
        """
        with torch.no_grad():
            obs_input = obs.unsqueeze(0).to(self.device)  # [1, obs_dim]
            mask_input = mask.unsqueeze(0).to(self.device)  # [1, n_rbg, n_actions]
            
            # =====================================================
            # DEBUG: Verify data flows through model (EVERY TTI, layer 0)
            # =====================================================
            if self.verbose and layer_idx == 0:
                print(f"\n{'='*60}")
                print(f"[MODEL INPUT] TTI {self.total_steps} Layer {layer_idx} - Verification:")
                print(f"{'='*60}")
                print(f"  obs_input shape: {obs_input.shape}")
                print(f"  obs_input device: {obs_input.device}")
                print(f"  obs_input[:10]: {obs_input[0,:10].cpu().tolist()}")
                print(f"  obs_input non-zero count: {(obs_input != 0).sum().item()}/{obs_input.numel()}")
                print(f"  mask_input shape: {mask_input.shape}")
                print(f"  mask_input valid actions per RBG: {mask_input[0].sum(dim=-1).cpu().tolist()[:5]}...")
            
            # Get logits from actor using forward_all()
            logits = self.actor.forward_all(obs_input)  # [1, n_rbg, n_actions]
            
            # =====================================================
            # DEBUG: Model output before masking (EVERY TTI, layer 0)
            # =====================================================
            if self.verbose and layer_idx == 0:
                print(f"\n  [MODEL OUTPUT] Raw logits:")
                print(f"    logits shape: {logits.shape}")
                print(f"    logits[0,0,:5] (RBG 0, first 5 actions): {logits[0,0,:5].cpu().tolist()}")
                print(f"    logits[0,0,:] range: [{logits[0,0,:].min().item():.4f}, {logits[0,0,:].max().item():.4f}]")
                print(f"    logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
            
            # Apply mask
            logits = apply_action_mask_to_logits(logits, mask_input)
            
            if explore and np.random.rand() < epsilon:
                # Epsilon-greedy: random valid action
                actions = []
                for rbg in range(mask_input.shape[1]):
                    valid_actions = torch.where(mask_input[0, rbg])[0]
                    action = valid_actions[torch.randint(len(valid_actions), (1,))]
                    actions.append(action.item())
                actions = torch.tensor(actions, dtype=torch.long)
                action_source = "RANDOM (epsilon-greedy)"
            else:
                # Greedy: argmax
                probs = torch.softmax(logits, dim=-1)
                actions = torch.argmax(probs, dim=-1).squeeze(0)  # [n_rbg]
                action_source = "GREEDY (argmax)"
            
            # =====================================================
            # DEBUG: Final actions (EVERY TTI, layer 0)
            # =====================================================
            if self.verbose and layer_idx == 0:
                print(f"\n  [ACTIONS] {action_source}:")
                print(f"    actions shape: {actions.shape}")
                print(f"    actions[:10]: {actions[:10].tolist()}")
                unique_actions = torch.unique(actions).tolist()
                print(f"    unique actions: {unique_actions}")
                print(f"{'='*60}\n")
        
        return actions
    
    def generate_allocation_matrix(self, obs, features, max_ues, max_layers, num_rbg, 
                                   eligible_ues, explore=True,
                                   num_subbands=1, num_rbs=None,
                                   rbg_size=None, subband_size=None,
                                   mu_corr_threshold=0.9, min_cqi=1,
                                   cross_corr=None):
        """
        Paper-style 16-layer scheduling:
          - Loop 16 spatial layers
          - Per-layer update allocated_RBG feature (alloc_ratio col=2)
          - Table 4 constraints per RBG: |L|=8, rank=2, total layers=16
        """
        # Output: [num_rbg][max_layers] UE id (1-indexed) or 0
        allocation_matrix = [[0 for _ in range(max_layers)] for _ in range(num_rbg)]
        
        # Track allocation per layer for observation building (0-based UE indices)
        # prev_alloc[layer][rbg] = UE index (0-based) or max_ues (NO_ALLOC)
        prev_alloc_by_layer = []

        # Make a dynamic copy of features to update alloc_ratio per layer
        features_dyn = [row[:] for row in features]  # list of lists

        # Extract raw buffer status from column 3 (still raw bytes here; mask uses >0 only)
        buffer_status = [features_dyn[u][3] for u in range(max_ues)]

        # ---- Track per-RBG state (Table 4) ----
        # rank_used_per_rbg_ue[rbg][ue] in {0,1,2}
        rank_used_per_rbg_ue = [[0 for _ in range(max_ues)] for _ in range(num_rbg)]
        # ue_set_per_rbg[rbg] = set of UEs (0-based) already on this RBG
        ue_set_per_rbg = [set() for _ in range(num_rbg)]
        # total layers used on each RBG
        layers_used_per_rbg = [0 for _ in range(num_rbg)]
        # last_layer_per_rbg_ue[rbg][ue] = last layer index assigned to UE on this RBG (-1 if none)
        # Used for continuous layer constraint (single MAC PDU per UE)
        last_layer_per_rbg_ue = [[-1 for _ in range(max_ues)] for _ in range(num_rbg)]

        # ---- Track allocated RBG count per UE across the whole bandwidth (paper d_u) ----
        # allocated_rbg[u][rbg] = 1 if UE u scheduled on that RBG in any previous layer
        allocated_rbg = [[0 for _ in range(num_rbg)] for _ in range(max_ues)]

        all_actions, all_masks, all_obs_layers = [], [], []

        for layer_idx in range(max_layers):
            # ---- Update allocated_RBG feature (alloc_ratio col=2) BEFORE choosing this layer ----
            # d_u = number of RBGs allocated to UE u so far (not streams)
            dmax = float(num_rbg) if num_rbg > 0 else 1.0
            for u in range(max_ues):
                d_u = sum(allocated_rbg[u])      # count allocated RBGs
                features_dyn[u][2] = d_u / dmax  # hat_d_u in [0,1]

            # Build obs for this layer with cross_corr and prev_alloc for max_corr_feat
            obs_layer = self.build_observation(
                features_dyn, max_ues, num_rbg,
                allocated_rbg=allocated_rbg,
                cross_corr=cross_corr,
                prev_alloc=prev_alloc_by_layer,
                layer=layer_idx
            )
            all_obs_layers.append(obs_layer)  # Store for layer-step transitions

            # Build mask for this layer (Table 4 + continuous layer constraint + pack from layer 0)
            hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
            max_paired_ues = hp.get('max_paired_ues', 4)  # Default to 4 UEs max per RBG
            mask = self.build_action_mask(
                num_rbg=num_rbg,
                max_ues=max_ues,
                eligible_ues=eligible_ues,
                buffer_status=buffer_status,
                ue_set_per_rbg=ue_set_per_rbg,
                rank_used_per_rbg_ue=rank_used_per_rbg_ue,
                max_ue_layers_per_rbg=max_paired_ues,  # Use configurable max paired UEs
                max_rank_per_ue_per_rbg=2,
                last_layer_per_rbg_ue=last_layer_per_rbg_ue,
                current_layer_idx=layer_idx,
                layers_used_per_rbg=layers_used_per_rbg,
                features=features_dyn,
                num_subbands=num_subbands,
                num_rbs=num_rbs,
                rbg_size=rbg_size,
                subband_size=subband_size,
                mu_corr_threshold=mu_corr_threshold,
                min_cqi=min_cqi
            )

            # Select actions for all RBGs
            if self.training_enabled and self.actor is not None:
                epsilon = max(0.1, 1.0 - self.layer_steps / 30000)  # Use layer_steps for epsilon decay (adjusted for 100 frames)
                actions = self.select_action(obs_layer, mask, explore=explore, epsilon=epsilon, layer_idx=layer_idx)
            else:
                actions = []
                for rbg in range(num_rbg):
                    valid_actions = torch.where(mask[rbg])[0]
                    action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                    actions.append(action)
                actions = torch.tensor(actions, dtype=torch.long)

            all_actions.append(actions)
            all_masks.append(mask)

            # Apply actions + update trackers
            # NO_ALLOC is always the last action in the fixed action space
            if self.fixed_n_actions is not None:
                NO_ALLOC = self.fixed_n_actions - 1
            else:
                NO_ALLOC = max_ues
            layer_allocs = []  # Track allocations for this layer
            for rbg in range(num_rbg):
                a = int(actions[rbg].item())
                if a == NO_ALLOC or a >= max_ues:
                    layer_allocs.append(f"RBG{rbg}:--")
                    continue

                ue0 = a           # 0-based UE index
                ue_id = ue0 + 1   # 1-based for MATLAB

                # Safety checks
                if ue_id not in eligible_ues:
                    layer_allocs.append(f"RBG{rbg}:X(inelig)")
                    continue
                if buffer_status[ue0] <= 0:
                    layer_allocs.append(f"RBG{rbg}:X(buf=0)")
                    continue
                if layers_used_per_rbg[rbg] >= max_layers:
                    layer_allocs.append(f"RBG{rbg}:X(full)")
                    continue

                # Table 4: distinct UE <= 8, rank <= 2
                if ue0 in ue_set_per_rbg[rbg]:
                    if rank_used_per_rbg_ue[rbg][ue0] >= 2:
                        layer_allocs.append(f"RBG{rbg}:X(rank2)")
                        continue
                else:
                    if len(ue_set_per_rbg[rbg]) >= 8:
                        layer_allocs.append(f"RBG{rbg}:X(8UEs)")
                        continue
                    ue_set_per_rbg[rbg].add(ue0)

                # Commit allocation
                allocation_matrix[rbg][layer_idx] = ue_id
                layers_used_per_rbg[rbg] += 1
                rank_used_per_rbg_ue[rbg][ue0] += 1
                layer_allocs.append(f"RBG{rbg}:UE{ue_id}")
                
                # Update last layer tracker for continuous layer constraint
                last_layer_per_rbg_ue[rbg][ue0] = layer_idx

                # Update allocated RBG tracker (paper d_u)
                allocated_rbg[ue0][rbg] = 1
            
            # Track this layer's allocation for building next layer's observation (0-based UE indices)
            layer_alloc_0based = []
            for rbg in range(num_rbg):
                ue_id = allocation_matrix[rbg][layer_idx]
                if ue_id > 0:
                    layer_alloc_0based.append(ue_id - 1)  # Convert to 0-based
                else:
                    layer_alloc_0based.append(max_ues)  # NO_ALLOC
            prev_alloc_by_layer.append(layer_alloc_0based)
            
            # === LOG LAYER-BY-LAYER ALLOCATION ===
            if self.verbose:
                # Count valid allocations for this layer
                valid_allocs = [a for a in layer_allocs if ":UE" in a]
                no_allocs = [a for a in layer_allocs if ":--" in a]
                rejected = [a for a in layer_allocs if ":X(" in a]
                
                # Compact view: show UE assignments
                ue_assignments = [a.split(":")[1] for a in layer_allocs if ":UE" in a or ":--" in a]
                compact = " ".join([f"{allocation_matrix[rbg][layer_idx]}" for rbg in range(num_rbg)])
                
                print(f"  Layer {layer_idx:2d}: [{compact}] | alloc={len(valid_allocs)}, noop={len(no_allocs)}, reject={len(rejected)}")

        # ===== LOG CONTINUOUS LAYER ALLOCATION =====
        if self.verbose:
            print(f"\n  === CONTINUOUS LAYER ALLOCATION LOG ===")
            for rbg in range(num_rbg):
                # Collect UE -> layers mapping for this RBG
                ue_layers_map = {}  # {ue_id: [list of layer indices]}
                for layer_idx in range(max_layers):
                    ue_id = allocation_matrix[rbg][layer_idx]
                    if ue_id > 0:
                        if ue_id not in ue_layers_map:
                            ue_layers_map[ue_id] = []
                        ue_layers_map[ue_id].append(layer_idx)
                
                if ue_layers_map:
                    # Check continuity and format output
                    parts = []
                    for ue_id, layers in sorted(ue_layers_map.items()):
                        # Check if layers are continuous
                        is_continuous = True
                        for i in range(1, len(layers)):
                            if layers[i] != layers[i-1] + 1:
                                is_continuous = False
                                break
                        
                        layer_str = f"{layers}" if len(layers) > 1 else f"[{layers[0]}]"
                        status = "✓" if is_continuous else "✗ NON-CONTINUOUS!"
                        parts.append(f"UE{ue_id}:{layer_str}{status}")
                    
                    num_ues = len(ue_layers_map)
                    pairing_status = f"({num_ues} UEs paired)" if num_ues > 1 else "(single UE)"
                    print(f"    RBG {rbg:2d}: {', '.join(parts)} {pairing_status}")

        return allocation_matrix, all_actions, all_masks, all_obs_layers
    
    def start_server(self):
        """Start TCP server and handle MATLAB connection."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        
        print(f"[Trainer] DRL Training Server listening on port {self.port}")
        print(f"[Trainer] Waiting for MATLAB SchedulerDRL to connect...")
        print(f"[Trainer] Run MU_MIMO.m in MATLAB now")
        print(f"[Trainer] Press Ctrl+C to stop (checkpoint will be saved)\n")
        
        try:
            while True:
                self.client_socket, addr = self.server_socket.accept()
                print(f"[Trainer] MATLAB connected from {addr}")
                
                self.handle_training_session()
                
        except KeyboardInterrupt:
            print("\n[Trainer] Ctrl+C - Shutting down...")
            self._emergency_save("final_ctrl_c")
        finally:
            self.cleanup()
    
    def handle_training_session(self):
        """Handle one training session with MATLAB."""
        buffer = b""
        prev_obs_layers = None
        prev_actions_layers = None
        prev_masks_layers = None
        
        try:
            while True:
                try:
                    chunk = self.client_socket.recv(8192)
                    if not chunk:
                        print("[Trainer] MATLAB disconnected")
                        break
                    
                    buffer += chunk
                    
                    # Process complete messages
                    while b'\n' in buffer:
                        message, buffer = buffer.split(b'\n', 1)
                        
                        if not message.strip():
                            continue
                        
                        try:
                            data = json.loads(message.decode('utf-8'))
                        except json.JSONDecodeError as e:
                            print(f"[Trainer] JSON decode error: {e}")
                            continue
                        
                        # Handle TTI_OBS message
                        if data.get('type') == 'TTI_OBS':
                            self.handle_tti_observation(data, prev_obs_layers, prev_actions_layers, prev_masks_layers)
                            prev_obs_layers, prev_actions_layers, prev_masks_layers = self.process_tti(data)
                        
                except ConnectionResetError:
                    print("[Trainer] Connection reset by MATLAB")
                    break
                    
        except KeyboardInterrupt:
            print("\n[Trainer] Ctrl+C detected - saving checkpoint before exit...")
            self._emergency_save("keyboard_interrupt")
            raise  # Re-raise to propagate to start_server
            
        except Exception as e:
            print(f"[Trainer] Error: {e}")
            import traceback
            traceback.print_exc()
            print("[Trainer] Saving checkpoint due to error...")
            self._emergency_save("error")
        
        finally:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
    
    def _emergency_save(self, reason="emergency"):
        """Save checkpoint in emergency situations (error, Ctrl+C, etc.)."""
        if self.actor is None:
            print("[Trainer] No model to save (actor not initialized)")
            return
        
        hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
        out_dir = hp.get('out_dir', 'checkpoints')
        os.makedirs(out_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(out_dir, f"checkpoint_{reason}_tti{self.total_steps}_{timestamp}.pt")
        
        # DSACDUpdater doesn't have state_dict, save components separately
        updater_state = {}
        if self.updater is not None:
            try:
                updater_state = {
                    'q1': self.updater.q1.state_dict(),
                    'q2': self.updater.q2.state_dict(),
                    'q1_target': self.updater.q1_t.state_dict(),
                    'q2_target': self.updater.q2_t.state_dict(),
                    'log_alpha': self.updater.log_alpha.detach().cpu(),
                    'opt_actor': self.updater.opt_actor.state_dict(),
                    'opt_critic': self.updater.opt_critic.state_dict(),
                    'opt_alpha': self.updater.opt_alpha.state_dict(),
                }
            except Exception as e:
                print(f"[Trainer] Warning: Could not save updater state: {e}")
        
        torch.save({
            'actor': self.actor.state_dict() if self.actor else {},
            'updater': updater_state,
            'total_steps': self.total_steps,
            'layer_steps': self.layer_steps,
            'episode_count': self.episode_count,
        }, checkpoint_path)
        
        print(f"[Trainer] Emergency checkpoint saved to {checkpoint_path}")
        print(f"[Trainer] Stats at save: TTI={self.total_steps}, LayerSteps={self.layer_steps}")
    
    def handle_tti_observation(self, data, prev_obs_layers, prev_actions_layers, prev_masks_layers=None):
        if prev_obs_layers is None or prev_actions_layers is None:
            return

        max_layers = len(prev_actions_layers)

        # ---- IMPORTANT FIX: allocation for reward must be from the TTI we are storing (t-1) ----
        # When we receive TTI t observation, MATLAB is reporting metrics for TTI t-1 (last sent allocation).
        prev_alloc = self.last_allocation_matrix if self.last_allocation_matrix is not None else []

        # Per-layer rewards for PREVIOUS TTI (t-1)
        layer_rewards = self.calculate_layer_rewards(data, prev_alloc, max_layers)

        # Next state's first-layer observation for TTI t
        max_ues = data.get("max_ues", 16)
        num_rbg = data.get("num_rbg", 18)
        
        # Support both old and new observation formats
        if 'features' in data:
            features = data.get("features", [])
            curr_first_obs = self.build_observation(features, max_ues, num_rbg)
        else:
            # New format: pass entire data dict
            curr_first_obs = self.build_observation(data, max_ues, num_rbg)

        # Store one transition per layer
        for layer_idx in range(max_layers):
            state = prev_obs_layers[layer_idx]
            action = prev_actions_layers[layer_idx]  # shape [n_rbg]
            reward = float(layer_rewards[layer_idx])

            if layer_idx < max_layers - 1:
                next_state = prev_obs_layers[layer_idx + 1]
                next_mask = prev_masks_layers[layer_idx + 1] if prev_masks_layers else None
            else:
                next_state = curr_first_obs
                next_mask = None  # Will be filled from next TTI

            done = False
            
            # Get current mask for this layer
            curr_mask = prev_masks_layers[layer_idx] if prev_masks_layers else None

            self.replay_buffer.push(state, action, reward, next_state, done, curr_mask, next_mask)

            self.current_episode_reward += reward
            self.current_episode_length += 1
            self.layer_steps += 1

        # Verbose
        if self.verbose and (self.total_steps % 5 == 0):
            print(f"[Trainer] === LAYER-STEP EXPERIENCE ===")
            print(f"  Stored {max_layers} transitions (1 per layer)")
            print(f"  Total reward: {sum(layer_rewards):.4f}")
            print(f"  Layer rewards: min={min(layer_rewards):.4f}, max={max(layer_rewards):.4f}")
            # Show per-layer reward detail for first 6 layers
            non_zero_rewards = [(i, r) for i, r in enumerate(layer_rewards) if abs(r) > 0.01]
            if non_zero_rewards:
                detail = ", ".join([f"L{i}:{r:.3f}" for i, r in non_zero_rewards[:6]])
                print(f"  Per-layer detail: {detail}{'...' if len(non_zero_rewards) > 6 else ''}")
            print(f"  Buffer: {len(self.replay_buffer)}/{self.replay_buffer.buffer.maxlen}")
            print(f"  Episode: reward={self.current_episode_reward:.2f}, length={self.current_episode_length} layer-steps")
            print(f"  Layer steps: {self.layer_steps} (TTI steps: {self.total_steps})")

        # Log evaluation statistics for this TTI
        self.log_eval_stats(data, prev_alloc, policy_type="sample")

        # Train every N layer-steps
        hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
        learning_starts = int(hp.get("learning_starts", 1000))
        train_freq = int(hp.get("train_freq", 4))
        batch_size = int(hp.get("batch_size", 256))

        if len(self.replay_buffer) >= learning_starts and (self.layer_steps % train_freq == 0):
            self.train_step(batch_size)
    
    def process_tti(self, data):
        """Process one TTI observation and return allocation."""
        self.total_steps += 1
        
        max_ues = data.get('max_ues', 16)
        max_layers = data.get('max_layers', 16)
        max_layers_per_ue = data.get('max_layers_per_ue', 2)
        num_rbg = data.get('num_rbg', 18)
        num_subbands = data.get('num_subbands', num_rbg)  # Default to num_rbg if not specified
        num_rbs = data.get('num_rbs', num_rbg)
        rbg_size = data.get('rbg_size', max(1, int(math.ceil(num_rbs / max(num_rbg, 1)))))
        subband_size = data.get('subband_size', max(1, int(math.ceil(num_rbs / max(num_subbands, 1)))))
        mu_corr_threshold = data.get('mu_corr_threshold', 0.9)
        min_cqi = data.get('min_cqi', 1)
        eligible_ues = data.get('eligible_ues', [])
        
        # =====================================================
        # LOG MATLAB DATA - Check if 7 features are received
        # =====================================================
        if self.verbose:  # Log EVERY TTI
            print(f"\n{'='*60}")
            print(f"[MATLAB DATA] TTI {self.total_steps} - Received fields from MATLAB:")
            print(f"{'='*60}")
            
            # Check what fields MATLAB sent
            matlab_fields = {
                'avg_throughput': data.get('avg_throughput'),
                'rank': data.get('rank'),
                'buffer': data.get('buffer'),
                'wideband_cqi': data.get('wideband_cqi'),
                'subband_cqi': data.get('subband_cqi'),
                'cross_corr': data.get('cross_corr'),
                'features': data.get('features'),  # Old format
            }
            
            for field, value in matlab_fields.items():
                if value is not None:
                    if isinstance(value, (list, np.ndarray)):
                        arr = np.array(value)
                        print(f"  ✓ {field:15s}: shape={arr.shape}, dtype={arr.dtype}")
                        if arr.size > 0 and arr.ndim <= 2:
                            if arr.ndim == 1:
                                # Show first few values
                                print(f"                    values[:5] = {arr[:5].tolist()}")
                                print(f"                    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                            else:
                                print(f"                    [0,:5] = {arr[0,:5].tolist() if arr.shape[1] >= 5 else arr[0].tolist()}")
                    else:
                        print(f"  ✓ {field:15s}: {value}")
                else:
                    print(f"  ✗ {field:15s}: NOT SENT")
            
            # Summary
            expected_fields = ['avg_throughput', 'rank', 'buffer', 'wideband_cqi', 'subband_cqi', 'cross_corr']
            received = sum(1 for f in expected_fields if data.get(f) is not None)
            print(f"\n  SUMMARY: Received {received}/6 raw fields from MATLAB")
            print(f"  (Python computes: norm_allocated_rbgs, max_corr_feat per layer)")
            print(f"{'='*60}\n")
        
        # Support both old 'features' array format and new separate fields format
        if 'features' in data:
            features = data.get('features', [])
            use_new_format = False
        else:
            # New format: build features from separate fields for backward compatibility
            avg_throughput = np.array(data.get('avg_throughput', [0] * max_ues), dtype=np.float32).flatten()
            rank_ue = np.array(data.get('rank', [1] * max_ues), dtype=np.float32).flatten()
            buffer_status = np.array(data.get('buffer', [0] * max_ues), dtype=np.float32).flatten()
            wideband_cqi = np.array(data.get('wideband_cqi', [7] * max_ues), dtype=np.float32).flatten()
            subband_cqi = np.array(data.get('subband_cqi', np.zeros((max_ues, num_rbg))), dtype=np.float32)
            
            # Build features array for backward compatibility with generate_allocation_matrix
            # features[u] = [throughput, rank, alloc_ratio(0), buffer, wideband_cqi, ...]
            features = []
            for u in range(max_ues):
                ue_features = [
                    avg_throughput[u] if u < len(avg_throughput) else 0,
                    rank_ue[u] if u < len(rank_ue) else 1,
                    0,  # alloc_ratio placeholder (updated per layer)
                    buffer_status[u] if u < len(buffer_status) else 0,
                    wideband_cqi[u] if u < len(wideband_cqi) else 7,
                ]
                # Add subband CQI per RBG
                if subband_cqi.ndim == 2 and u < subband_cqi.shape[0]:
                    for rbg in range(num_rbg):
                        if rbg < subband_cqi.shape[1]:
                            ue_features.append(subband_cqi[u, rbg])
                        else:
                            ue_features.append(wideband_cqi[u] if u < len(wideband_cqi) else 7)
                else:
                    for rbg in range(num_rbg):
                        ue_features.append(wideband_cqi[u] if u < len(wideband_cqi) else 7)
                features.append(ue_features)
            use_new_format = True
        
        # Ensure eligible_ues is always a list (MATLAB may send single int)
        if isinstance(eligible_ues, (int, float)):
            eligible_ues = [int(eligible_ues)] if eligible_ues > 0 else []
        elif not isinstance(eligible_ues, list):
            eligible_ues = list(eligible_ues) if eligible_ues else []
        
        if self.verbose and self.total_steps % 10 == 1:
            frame = data.get('frame', 0)
            slot = data.get('slot', 0)
            print(f"\n[Trainer] Step {self.total_steps} | Frame {frame}, Slot {slot}")
            print(f"  Eligible UEs: {len(eligible_ues)}/{max_ues}")
            if self.replay_buffer is not None:
                print(f"  Buffer size: {len(self.replay_buffer)}")
            if len(self.episode_rewards) > 0:
                print(f"  Avg reward (last 10 eps): {np.mean(self.episode_rewards[-10:]):.2f}")
            if use_new_format:
                print(f"  Using new observation format with separate fields")
        
        # Build observation - use data dict for new format, features for old format
        if use_new_format:
            obs = self.build_observation(data, max_ues, num_rbg)
        else:
            obs = self.build_observation(features, max_ues, num_rbg)
        
        if self.verbose and self.total_steps % 10 == 1:
            print(f"[Trainer] === OBSERVATION ===")
            print(f"  Shape: {obs.shape}")
            print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}, Mean: {obs.mean():.4f}")
            print(f"  Sample values (first 10): {obs[:10].tolist()}")
        
        # Initialize networks if not done yet
        if self.actor is None:
            obs_dim = obs.shape[0]
            n_actions = max_ues + 1  # UEs + NO_ALLOC
            hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
            self.initialize_networks(
                obs_dim=obs_dim,
                n_rbg=num_rbg,
                n_actions_per_rbg=n_actions,
                hidden_dim=hp.get('hidden_dim', 256),
                n_quantiles=hp.get('n_quantiles', 32),
                gamma=hp.get('gamma', 0.99),
                tau=hp.get('tau', 0.005),
                lr_actor=hp.get('lr_actor', 3e-4),
                lr_critic=hp.get('lr_critic', 3e-4),
                lr_alpha=hp.get('lr_alpha', 3e-4),
                beta=hp.get('beta', 0.98),
                fallback_action=hp.get('fallback_action', -1)
            )
        
        # Generate allocation matrix
        explore = self.training_enabled
        
        # Extract cross_corr matrix if available (new format)
        cross_corr = None
        if use_new_format and 'cross_corr' in data:
            cross_corr = np.array(data.get('cross_corr', []), dtype=np.float32)
            if cross_corr.ndim != 3 or cross_corr.shape[0] < max_ues:
                cross_corr = None  # Invalid shape, don't use
        
        allocation_matrix, all_actions, all_masks, all_obs_layers = self.generate_allocation_matrix(
            obs=obs,
            features=features,
            max_ues=max_ues,
            max_layers=max_layers,
            num_rbg=num_rbg,
            eligible_ues=eligible_ues,
            explore=explore,
            num_subbands=num_subbands,
            num_rbs=num_rbs,
            rbg_size=rbg_size,
            subband_size=subband_size,
            mu_corr_threshold=mu_corr_threshold,
            min_cqi=min_cqi,
            cross_corr=cross_corr
        )
        
        # Log allocation statistics
        if self.verbose:
            total_allocs = sum(1 for rbg_row in allocation_matrix for ue in rbg_row if ue > 0)
            unique_ues = set(ue for rbg_row in allocation_matrix for ue in rbg_row if ue > 0)
            epsilon = max(0.1, 1.0 - self.layer_steps / 50000) if self.training_enabled else 0.0
            
            print(f"[Trainer] === ACTIONS ===")
            print(f"  Exploration: epsilon={epsilon:.4f} (layer_steps={self.layer_steps})")
            print(f"  Total allocations: {total_allocs}/{num_rbg * max_layers} slots")
            print(f"  Scheduled UEs: {sorted(unique_ues)}")
            print(f"  Unique UEs scheduled: {len(unique_ues)}/{len(eligible_ues)}")
            
            # Show layer-by-layer statistics
            layer_allocations = [sum(1 for rbg_row in allocation_matrix if rbg_row[l] > 0) for l in range(max_layers)]
            print(f"  Allocations per layer: {layer_allocations}")
            
            # Show first 5 RBGs with all layers
            print(f"  Sample allocation (first 5 RBGs, all 16 layers):")
            for rbg in range(min(5, num_rbg)):
                scheduled = [ue for ue in allocation_matrix[rbg] if ue > 0]
                if scheduled:
                    ue_counts = {ue: allocation_matrix[rbg].count(ue) for ue in set(scheduled)}
                    print(f"    RBG {rbg:2d}: {ue_counts} (total: {len(scheduled)} layers)")
                else:
                    print(f"    RBG {rbg:2d}: empty")
        
        # Send to MATLAB
        self.send_allocation(allocation_matrix)
        
        # Shift allocation tracking: last -> prev, then store new as last
        # prev_allocation_matrix = allocation from TTI t-1 (used for rewards)
        # last_allocation_matrix = allocation from TTI t (just sent to MATLAB)
        self.prev_allocation_matrix = self.last_allocation_matrix
        self.last_allocation_matrix = allocation_matrix
        
        if self.verbose:
            print(f"[Trainer] ✓ Sent allocation matrix to MATLAB\n")
        
        # Save checkpoint at specific TTI milestones
        hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
        save_at_ttis = hp.get('save_at_ttis', [1950])  # Default: save at 1950 TTI
        save_every_n_tti = hp.get('save_every_n_tti', 10)  # Save every N TTIs
        
        # Save at specific milestones
        if self.total_steps in save_at_ttis and self.actor is not None:
            out_dir = hp.get('out_dir', 'checkpoints')
            print(f"[Trainer] Milestone TTI {self.total_steps} reached - saving checkpoint")
            self.save_checkpoint(out_dir)
        
        # Save periodically every N TTIs
        if save_every_n_tti > 0 and self.total_steps % save_every_n_tti == 0 and self.actor is not None:
            out_dir = hp.get('out_dir', 'checkpoints')
            print(f"[Trainer] Periodic save at TTI {self.total_steps}")
            self.save_checkpoint(out_dir)
        
        # Return for layer-step experience storage (include masks for DSACD training)
        return all_obs_layers, all_actions, all_masks
    
    def _rate_mbps(self, mcs_idx, n_prb, tti_ms=1.0, n_symb=14, overhead_re_per_prb=18):
        """Compute rate in Mbps for one layer on given RBG."""
        tbs_bytes = tbs_38214_bytes(mcs_idx, n_prb, n_symb=n_symb, overhead_re_per_prb=overhead_re_per_prb)
        duration_s = tti_ms / 1000.0
        return (tbs_bytes * 8.0) / 1e6 / max(duration_s, 1e-9)
    
    def _compute_set_tput(self, alloc_set, rbg_idx, cross_corr, subband_mcs, buffer_status, 
                          max_ues, prbs_per_rbg=18, tti_ms=1.0):
        """
        Compute throughput for a set of UEs on RBG m with cross-correlation penalty.
        Matches toy5g_env_adapter.compute_set_tput().
        
        Args:
            alloc_set: list of 1-indexed UE IDs [ue1, ue2, ...]
            rbg_idx: RBG index (0-based)
            cross_corr: [U, U, M] cross-correlation matrix from MATLAB
            subband_mcs: [U, M] subband MCS indexed by 0-based UE
            buffer_status: [U] buffer status indexed by 0-based UE  
            max_ues: total number of UEs
            prbs_per_rbg: PRBs per RBG
            tti_ms: TTI duration in ms
        """
        if len(alloc_set) == 0:
            return 0.0
        
        # Find max cross-correlation between any pair of UEs scheduled on this RBG
        max_corr = 0.0
        if len(alloc_set) > 1:
            for i in range(len(alloc_set) - 1):
                u1 = alloc_set[i] - 1  # Convert to 0-indexed
                for j in range(i + 1, len(alloc_set)):
                    u2 = alloc_set[j] - 1  # Convert to 0-indexed
                    if 0 <= u1 < max_ues and 0 <= u2 < max_ues:
                        if cross_corr.shape[2] > rbg_idx:
                            corr = float(cross_corr[u1, u2, rbg_idx])
                            max_corr = max(max_corr, corr)
        
        # Penalty based on correlation - divides throughput
        penalty = (1.0 - max_corr) / max(len(alloc_set), 1)
        
        # Sum throughput for all UEs in set (only if buffer > 0)
        tput = 0.0
        for ue_id in alloc_set:
            u = ue_id - 1  # Convert to 0-indexed
            if 0 <= u < max_ues and buffer_status[u] > 0:
                if rbg_idx < subband_mcs.shape[1]:
                    mcs = int(subband_mcs[u, rbg_idx])
                    tput += self._rate_mbps(mcs, prbs_per_rbg, tti_ms=tti_ms)
        
        return tput * penalty

    def calculate_layer_rewards(self, current_data, allocation_matrix, max_layers):
        """
        Calculate per-layer rewards matching toy5g_env_adapter.new_reward_compute().
        
        Paper formula (Eq. 21):
            raw_all[u] = (T_cur - T_prev) / buffer[u]
            reward[m] = clamp(raw_all[chosen] / max_raw, -1, 1)
            layer_reward = sum(reward[m] for m in range(n_rbg))
        
        Args:
            current_data: TTI observation data from MATLAB (should be for previous TTI)
            allocation_matrix: [num_rbg][max_layers] allocation from previous TTI
            max_layers: number of spatial layers
        
        Returns:
            layer_rewards: [max_layers] list of rewards
        """
        # If MATLAB provides pre-computed rewards, use them
        if 'prev_layer_rewards' in current_data:
            lr = current_data.get('prev_layer_rewards', [])
            if isinstance(lr, (list, tuple)) and len(lr) == max_layers:
                return list(lr)
        if 'layer_rewards' in current_data:
            lr = current_data.get('layer_rewards', [])
            if isinstance(lr, (list, tuple)) and len(lr) == max_layers:
                return list(lr)
        
        # No allocation matrix - return zeros
        if not allocation_matrix:
            return [0.0] * max_layers
        
        num_rbg = len(allocation_matrix)
        max_ues = current_data.get('max_ues', 16)
        noop = 0  # NOOP action = 0 (no allocation)
        eps = 1e-9
        
        # Get required data from MATLAB
        buffer_status = np.array(current_data.get('buffer', [1e6] * max_ues), dtype=np.float32).flatten()
        subband_cqi = np.array(current_data.get('subband_cqi', [[7]*num_rbg]*max_ues), dtype=np.float32)
        cross_corr = np.array(current_data.get('cross_corr', np.zeros((max_ues, max_ues, num_rbg))))
        
        # Convert CQI to MCS (approximate)
        if subband_cqi.ndim == 2:
            subband_mcs = np.zeros_like(subband_cqi, dtype=np.float32)
            for u in range(min(subband_cqi.shape[0], max_ues)):
                for m in range(min(subband_cqi.shape[1], num_rbg)):
                    cqi = int(np.clip(subband_cqi[u, m], 0, 15))
                    subband_mcs[u, m] = CQI_TO_MCS[cqi]
        else:
            subband_mcs = np.full((max_ues, num_rbg), 14, dtype=np.float32)
        
        # Ensure cross_corr has correct shape
        if cross_corr.ndim != 3 or cross_corr.shape != (max_ues, max_ues, num_rbg):
            cross_corr = np.zeros((max_ues, max_ues, num_rbg), dtype=np.float32)
        
        # Calculate per-layer rewards
        layer_rewards = [0.0] * max_layers
        
        for layer_idx in range(max_layers):
            layer_reward = 0.0
            
            for rbg_idx in range(num_rbg):
                # ---- Previous set (layers 0..layer_idx-1) ----
                prev_alloc = []
                if layer_idx > 0:
                    for l_prev in range(layer_idx):
                        ue_prev = int(allocation_matrix[rbg_idx][l_prev])
                        if ue_prev > 0 and ue_prev != noop:  # Valid UE (1-indexed, >0)
                            prev_alloc.append(ue_prev)
                
                T_prev = self._compute_set_tput(
                    prev_alloc, rbg_idx, cross_corr, subband_mcs, buffer_status, max_ues
                )
                
                # ---- Chosen action for this layer ----
                chosen = int(allocation_matrix[rbg_idx][layer_idx])
                
                if chosen == noop or chosen <= 0:
                    # NOOP selected - no throughput change
                    # If all raw rewards are negative (adding any UE reduces throughput),
                    # then NOOP is optimal and should get positive reward
                    # For simplicity, reward 0 for NOOP
                    continue
                
                # ---- Compute raw rewards for all candidates ----
                raw_all = np.zeros(max_ues + 1, dtype=np.float32)  # +1 for NOOP
                
                for u in range(1, max_ues + 1):  # 1-indexed UE IDs
                    u_idx = u - 1  # 0-indexed for array access
                    if buffer_status[u_idx] <= 0:
                        continue
                    
                    curr_alloc = prev_alloc + [u]
                    T_cur = self._compute_set_tput(
                        curr_alloc, rbg_idx, cross_corr, subband_mcs, buffer_status, max_ues
                    )
                    
                    # Paper formula: (T_cur - T_prev) / buffer
                    raw_all[u] = (T_cur - T_prev) / max(float(buffer_status[u_idx]), eps)
                
                # ---- Normalization (Eq. 21) ----
                max_raw = np.max(raw_all)
                
                if max_raw > 0:
                    # Normalize chosen action's reward
                    rbg_reward = np.clip(raw_all[chosen] / max_raw, -1.0, 1.0)
                elif max_raw < 0:
                    # All choices reduce throughput - chosen should be penalized
                    rbg_reward = -1.0
                else:
                    # All zero - neutral
                    rbg_reward = 0.0
                
                layer_reward += rbg_reward
            
            layer_rewards[layer_idx] = layer_reward
        
        return layer_rewards
    
    def log_eval_stats(self, data, allocation_matrix, policy_type="sample"):
        """
        Log evaluation statistics for the current TTI.
        
        Args:
            data: TTI observation data from MATLAB
            allocation_matrix: Current allocation [num_rbg x max_layers]
            policy_type: "sample" (current policy), "greedy", or "random"
        """
        if self.eval_log is None:
            return
        
        # Extract metrics from data (MATLAB should send these)
        tti = self.total_steps
        max_ues = data.get('max_ues', 16)
        
        # Get throughput from MATLAB feedback
        total_cell_tput = data.get('prev_tti_throughput', 0) or data.get('tti_throughput', 0) or 0
        
        # Per-UE throughputs - build list [ue0_tput, ue1_tput, ...]
        ue_tputs_dict = data.get('prev_ue_throughputs', {}) or data.get('ue_throughputs', {}) or {}
        total_ue_tput = []
        for ue_id in range(1, max_ues + 1):  # 1-indexed UE IDs
            tput = float(ue_tputs_dict.get(ue_id, ue_tputs_dict.get(str(ue_id), 0)))
            total_ue_tput.append(tput)
        
        # Per-UE allocation counts - build list [ue0_allocs, ue1_allocs, ...]
        alloc_counts = [0] * max_ues
        avg_layers_per_rbg = 0
        pf_utility = 0
        
        if allocation_matrix:
            num_rbg = len(allocation_matrix)
            max_layers = len(allocation_matrix[0]) if allocation_matrix else 16
            
            # Count allocations per UE
            for rbg_row in allocation_matrix:
                for ue in rbg_row:
                    if ue > 0 and ue <= max_ues:
                        alloc_counts[ue - 1] += 1  # Convert 1-indexed to 0-indexed
            
            # Average layers per RBG
            layers_per_rbg = [sum(1 for ue in rbg_row if ue > 0) for rbg_row in allocation_matrix]
            avg_layers_per_rbg = sum(layers_per_rbg) / num_rbg if num_rbg > 0 else 0
            
            # PF utility: sum of log(throughput) for all UEs with allocations
            pf_utility = 0
            for ue_idx, (tput, allocs) in enumerate(zip(total_ue_tput, alloc_counts)):
                if allocs > 0 and tput > 0:
                    pf_utility += math.log(max(tput, 1e-6))
                elif allocs > 0:
                    pf_utility += math.log(1e-6)  # Placeholder for UE with allocation but no throughput data
        
        # Store in eval_log
        if policy_type in self.eval_log:
            self.eval_log[policy_type]["tti"].append(tti)
            self.eval_log[policy_type]["total_cell_tput"].append(total_cell_tput)
            self.eval_log[policy_type]["total_ue_tput"].append(total_ue_tput)  # List per UE
            self.eval_log[policy_type]["alloc_counts"].append(alloc_counts)    # List per UE
            self.eval_log[policy_type]["pf_utility"].append(pf_utility)
            self.eval_log[policy_type]["avg_layers_per_rbg"].append(avg_layers_per_rbg)
        
        # Periodic logging
        if self.verbose and self.total_steps % 50 == 0:
            total_allocs = sum(alloc_counts)
            scheduled_ues = sum(1 for a in alloc_counts if a > 0)
            print(f"[Trainer] === EVAL STATS (TTI {tti}, {policy_type}) ===")
            print(f"  Total cell throughput: {total_cell_tput:.2f}")
            print(f"  Total UE throughput (sum): {sum(total_ue_tput):.2f}")
            print(f"  Total allocations: {total_allocs}, Scheduled UEs: {scheduled_ues}")
            print(f"  Avg layers/RBG: {avg_layers_per_rbg:.2f}")
            print(f"  PF utility: {pf_utility:.4f}")
    
    def train_step(self, batch_size=256):
        """Perform one training step using DSACD's per-RBG batch format.
        
        DSACD expects per-RBG experiences with keys:
          observation, next_observation: [B, obs_dim]
          rbg_index: [B] (0..NRBG-1)
          action: [B] (0..A-1)
          reward: [B]
          action_mask, next_action_mask: [B, A] bool
        
        We expand from layer-step experiences to per-RBG experiences.
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch - each sample is a layer-step with [n_rbg] actions
        states, actions, rewards, next_states, dones, masks, next_masks = self.replay_buffer.sample(batch_size)
        
        # Get dimensions
        n_rbg = actions.shape[1] if len(actions.shape) > 1 else 18
        
        # Use fixed network action dimension (critical for consistency)
        if self.fixed_n_actions is not None:
            n_actions = self.fixed_n_actions
        else:
            max_ues = self.hyperparams.get('max_ues', 16) if isinstance(self.hyperparams, dict) else 16
            n_actions = max_ues + 1  # +1 for NO_ALLOC
        
        # Randomly select one RBG per sample for training (paper approach)
        # This creates per-RBG experiences from layer-step experiences
        rbg_indices = torch.randint(0, n_rbg, (batch_size,))
        
        # Extract single action per sample based on randomly selected RBG
        single_actions = actions[torch.arange(batch_size), rbg_indices]
        
        # Handle masks - create default if not stored
        # IMPORTANT: Pad masks to fixed size n_actions (max_ues + 1) since
        # different TTIs may have different numbers of eligible UEs
        batch_masks = []
        batch_next_masks = []
        for i in range(batch_size):
            if masks[i] is not None:
                mask_i = masks[i][rbg_indices[i]]
                # Pad to n_actions if smaller
                if len(mask_i) < n_actions:
                    padded = torch.zeros(n_actions, dtype=torch.bool)
                    padded[:len(mask_i)] = mask_i
                    mask_i = padded
                elif len(mask_i) > n_actions:
                    mask_i = mask_i[:n_actions]
                batch_masks.append(mask_i)
            else:
                # Default: all actions valid
                default_mask = torch.ones(n_actions, dtype=torch.bool)
                batch_masks.append(default_mask)
            
            if next_masks[i] is not None:
                next_mask_i = next_masks[i][rbg_indices[i]]
                # Pad to n_actions if smaller
                if len(next_mask_i) < n_actions:
                    padded = torch.zeros(n_actions, dtype=torch.bool)
                    padded[:len(next_mask_i)] = next_mask_i
                    next_mask_i = padded
                elif len(next_mask_i) > n_actions:
                    next_mask_i = next_mask_i[:n_actions]
                batch_next_masks.append(next_mask_i)
            else:
                default_mask = torch.ones(n_actions, dtype=torch.bool)
                batch_next_masks.append(default_mask)
        
        batch_masks = torch.stack(batch_masks)
        batch_next_masks = torch.stack(batch_next_masks)
        
        # Build DSACD batch dict
        batch = {
            'observation': states.to(self.device),
            'next_observation': next_states.to(self.device),
            'rbg_index': rbg_indices.to(self.device),
            'action': single_actions.to(self.device),
            'reward': rewards.to(self.device),
            'action_mask': batch_masks.to(self.device),
            'next_action_mask': batch_next_masks.to(self.device),
        }
        
        # Update networks
        metrics = self.updater.update(batch)
        
        # Log metrics
        if self.layer_steps % 50 == 0:
            print(f"\n[Trainer] === TRAINING STEP {self.layer_steps} ===")
            print(f"  Batch size: {batch_size}")
            print(f"  TTI steps: {self.total_steps}")
            print(f"  Layer steps: {self.layer_steps}")
            print(f"  Loss Q: {metrics.get('loss_q', 0):.4f}")
            print(f"  Loss Actor: {metrics.get('loss_pi', 0):.4f}")
            print(f"  Alpha (temperature): {metrics.get('alpha', 0):.4f}")
            print(f"  Reward (batch mean): {rewards.mean():.4f}")
            print(f"  Reward (batch std): {rewards.std():.4f}")
            
            # Store training metrics
            if self.train_log is not None:
                self.train_log['tti'].append(int(self.total_steps))
                self.train_log['alpha'].append(float(metrics.get('alpha', 0)))
                self.train_log['loss_q'].append(float(metrics.get('loss_q', 0)))
                self.train_log['loss_pi'].append(float(metrics.get('loss_pi', 0)))
        
        # Save checkpoint periodically
        hp = self.hyperparams if isinstance(self.hyperparams, dict) else {}
        save_freq = hp.get('save_freq', 10000)
        if self.total_steps % save_freq == 0:
            out_dir = hp.get('out_dir', 'checkpoints')
            self.save_checkpoint(out_dir)
    
    def send_allocation(self, allocation_matrix):
        """Send allocation matrix to MATLAB."""
        response = {
            'type': 'TTI_ALLOC',
            'allocation': allocation_matrix
        }
        json_str = json.dumps(response) + '\n'
        self.client_socket.sendall(json_str.encode('utf-8'))
    
    def cleanup(self):
        """Clean up resources."""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        
        # Save final checkpoint
        if self.training_enabled and self.actor is not None:
            self.save_checkpoint()
            print(f"[Trainer] Final stats:")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Episodes: {self.episode_count}")
            if len(self.episode_rewards) > 0:
                print(f"  Mean episode reward: {np.mean(self.episode_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="DRL Training Server for MATLAB 5G Scheduler")
    
    # Server settings
    parser.add_argument('--port', type=int, default=5555, help='TCP port for MATLAB connection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose logging')
    
    # Network architecture
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--n_quantiles', type=int, default=32, help='Number of quantiles for distributional critic')
    
    # DSACD hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network soft update rate')
    parser.add_argument('--beta', type=float, default=0.98, help='Quantile Huber loss parameter')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--lr_alpha', type=float, default=3e-4, help='Temperature parameter learning rate')
    parser.add_argument('--fallback_action', type=int, default=-1, help='Fallback action index (last action)')
    
    # Training settings
    parser.add_argument('--rb_capacity', type=int, default=50000, help='Replay buffer capacity')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_starts', type=int, default=500, help='Steps before training starts')
    parser.add_argument('--train_freq', type=int, default=8, help='Training frequency (every N steps)')
    parser.add_argument('--save_freq', type=int, default=1000, help='Checkpoint save frequency')
    parser.add_argument('--max_paired_ues', type=int, default=4, help='Max distinct UEs per RBG (MU-MIMO pairing limit)')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint to continue training')
    parser.add_argument('--out_dir', type=str, default='outputs/matlab_training', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print(" " * 15 + "DRL Training Server for MATLAB 5G MU-MIMO Scheduler")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Port: {args.port}")
    print(f"Seed: {args.seed}")
    print(f"Hidden dim: {args.hidden}")
    print(f"N quantiles: {args.n_quantiles}")
    print(f"Gamma: {args.gamma}, Tau: {args.tau}, Beta: {args.beta}")
    print(f"Learning rates: actor={args.lr_actor}, critic={args.lr_critic}, alpha={args.lr_alpha}")
    print(f"Replay buffer: {args.rb_capacity}, Batch size: {args.batch_size}")
    print(f"Learning starts: {args.learning_starts}, Train freq: every {args.train_freq} steps")
    print(f"Save freq: every {args.save_freq} steps")
    print(f"Output dir: {args.out_dir}")
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
    print("=" * 70)
    print()
    
    # Create trainer with hyperparameters
    trainer = MATLABDRLTrainer(port=args.port, device=args.device, verbose=args.verbose)
    
    # Store hyperparameters for network initialization
    trainer.hyperparams = {
        'hidden_dim': args.hidden,
        'n_quantiles': args.n_quantiles,
        'gamma': args.gamma,
        'tau': args.tau,
        'beta': args.beta,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'lr_alpha': args.lr_alpha,
        'fallback_action': args.fallback_action,
        'rb_capacity': args.rb_capacity,
        'batch_size': args.batch_size,
        'learning_starts': args.learning_starts,
        'train_freq': args.train_freq,
        'save_freq': args.save_freq,
        'out_dir': args.out_dir,
        'max_paired_ues': args.max_paired_ues,  # Max distinct UEs per RBG
    }
    
    # Update replay buffer capacity
    trainer.replay_buffer = ReplayBuffer(capacity=args.rb_capacity)
    
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    trainer.start_server()


if __name__ == '__main__':
    main()