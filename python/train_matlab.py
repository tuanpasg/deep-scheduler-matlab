"""
train_matlab.py
===============
Training script sử dụng MATLAB làm môi trường thay vì toy5g.
Logic giống hệt train_2.py, chỉ khác:
  - env = MatlabEnvAdapter(...)  thay vì DeterministicToy5GEnvAdapter
  - Không cần eval_env (eval metrics do MATLAB tính, gửi qua TTI_DONE)
  - Metrics được log từ env._last_metrics

Cách dùng:
  # 1. Chạy Python server trước:
  python train_matlab.py --port 5555 --n_rbg 18 --n_layers 16 --ttis 5000

  # 2. Sau đó mở MATLAB và chạy simulation với:
  #    scheduler.TrainingMode = true;
  #    scheduler.connectToDRLAgent('127.0.0.1', 5555);
"""

import argparse
import os
import random
from collections import deque

import torch
import numpy as np

from matlab_env_adapter import MatlabEnvAdapter

from DSACD_multibranch import (
    MultiBranchActor,
    MultiBranchQuantileCritic,
    DSACDUpdater,
    DSACDHyperParams,
    ensure_nonempty_mask,
    apply_action_mask_to_logits,
)

from train_2_logging import (
    init_train_log,
    plot_training,
    save_logs,
)


# ─── Replay buffer (identical to train_2.py) ──────────────────────────────────

class SimpleReplay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, tr: dict):
        self.buf.append(tr)

    @property
    def size(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        batch = random.sample(self.buf, batch_size)

        def stack(key, dtype=None):
            xs = [b[key] for b in batch]
            x = torch.stack([t if t.ndim > 0 else t.view(1) for t in xs], dim=0)
            if dtype is not None:
                x = x.to(dtype)
            return x.to(device)

        out = {
            "observation":      stack("observation", torch.float32),
            "next_observation": stack("next_observation", torch.float32),
            "rbg_index":        stack("rbg_index", torch.long).squeeze(-1),
            "action":           stack("action", torch.long).squeeze(-1),
            "reward":           stack("reward", torch.float32).squeeze(-1),
            "action_mask":      stack("action_mask").to(torch.bool).squeeze(1)
                                if batch[0]["action_mask"].ndim == 1
                                else stack("action_mask").to(torch.bool),
            "next_action_mask": stack("next_action_mask").to(torch.bool).squeeze(1)
                                if batch[0]["next_action_mask"].ndim == 1
                                else stack("next_action_mask").to(torch.bool),
        }
        return out


# ─── Action sampling (identical to train_2.py) ───────────────────────────────

@torch.no_grad()
def sample_actions_for_layer(
    actor: MultiBranchActor,
    obs_layer: torch.Tensor,    # [obs_dim]
    masks_rbg: torch.Tensor,    # [NRBG, A] bool
    device: torch.device,
    fallback_action: int,
) -> torch.Tensor:
    obs_b = obs_layer.unsqueeze(0).to(device)          # [1, obs_dim]
    logits_all = actor.forward_all(obs_b).squeeze(0)   # [NRBG, A]

    masks_rbg = ensure_nonempty_mask(
        masks_rbg.to(device), fallback_action=fallback_action
    )
    logits_all = apply_action_mask_to_logits(logits_all, masks_rbg)

    dist = torch.distributions.Categorical(logits=logits_all)
    actions = dist.sample()   # [NRBG]
    return actions.cpu()


# ─── Eval log helpers ─────────────────────────────────────────────────────────

def init_matlab_eval_log() -> dict:
    return {
        "tti":                  [],
        "total_cell_tput":      [],
        "jain_throughput":      [],
        "pf_utility":           [],
        "avg_layers_per_rbg":   [],
        "no_schedule_rate":     [],
        "invalid_action_rate":  [],
    }


def append_matlab_eval(log: dict, tti: int, metrics: dict):
    log["tti"].append(tti)
    log["total_cell_tput"].append(float(metrics.get("total_cell_tput", 0.0)))
    log["jain_throughput"].append(float(metrics.get("jain_throughput", 0.0)))
    log["pf_utility"].append(float(metrics.get("pf_utility", 0.0)))
    log["avg_layers_per_rbg"].append(float(metrics.get("avg_layers_per_rbg", 0.0)))
    log["no_schedule_rate"].append(float(metrics.get("no_schedule_rate", 0.0)))
    log["invalid_action_rate"].append(float(metrics.get("invalid_action_rate", 0.0)))


def plot_matlab_eval(log: dict, out_path: str):
    """Simple plot of MATLAB-reported eval metrics over TTIs."""
    try:
        import matplotlib.pyplot as plt

        ttis = log["tti"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        pairs = [
            ("total_cell_tput",    "Cell Tput [Mbps]"),
            ("jain_throughput",    "Jain Fairness"),
            ("pf_utility",         "PF Utility"),
            ("avg_layers_per_rbg", "Avg Layers/RBG"),
            ("no_schedule_rate",   "NOOP Rate"),
            ("invalid_action_rate","Invalid Action Rate"),
        ]
        for ax, (key, title) in zip(axes.flat, pairs):
            ax.plot(ttis, log[key])
            ax.set_title(title)
            ax.set_xlabel("TTI")
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
    except ImportError:
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ── Environment ───────────────────────────────────────────────────────
    env = MatlabEnvAdapter(
        max_sched_ue=args.max_sched_ue,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        port=args.port,
        verbose=True,
    )

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print(f"[train_matlab] obs_dim={obs_dim}  act_dim={act_dim}  "
          f"n_rbg={env.n_rbg}  n_layers={env.n_layers}")

    # ── Networks ──────────────────────────────────────────────────────────
    hp = DSACDHyperParams(
        n_quantiles=args.n_quantiles,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        fallback_action=args.fallback_action,
    )

    actor = MultiBranchActor(
        obs_dim=obs_dim, n_rbg=env.n_rbg,
        act_dim=act_dim, hidden=args.hidden,
    )
    q1 = MultiBranchQuantileCritic(
        obs_dim=obs_dim, n_rbg=env.n_rbg,
        act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=args.hidden,
    )
    q2 = MultiBranchQuantileCritic(
        obs_dim=obs_dim, n_rbg=env.n_rbg,
        act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=args.hidden,
    )
    q1_t = MultiBranchQuantileCritic(
        obs_dim=obs_dim, n_rbg=env.n_rbg,
        act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=args.hidden,
    )
    q2_t = MultiBranchQuantileCritic(
        obs_dim=obs_dim, n_rbg=env.n_rbg,
        act_dim=act_dim, n_quantiles=hp.n_quantiles, hidden=args.hidden,
    )
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    updater = DSACDUpdater(
        actor=actor,
        q1=q1, q2=q2,
        q1_target=q1_t, q2_target=q2_t,
        n_rbg=env.n_rbg,
        act_dim=act_dim,
        hp=hp,
        device=str(device),
    )

    replay = SimpleReplay(capacity=args.rb_capacity)

    # ── Logs ──────────────────────────────────────────────────────────────
    train_log = init_train_log()
    eval_log  = init_matlab_eval_log()

    env.reset()

    # ── Training loop ─────────────────────────────────────────────────────
    for tti in range(args.ttis):
        env.begin_tti()

        for layer_ctx in env.layer_iter():
            actions_rbg = sample_actions_for_layer(
                actor=updater.actor,
                obs_layer=layer_ctx.obs,
                masks_rbg=layer_ctx.masks_rbg,
                device=device,
                fallback_action=args.fallback_action,
            )
            env.apply_layer_actions(layer_ctx, actions_rbg)
            transitions = env.compute_layer_transitions(layer_ctx)
            for tr in transitions:
                replay.add(tr)

        env.finish_tti()

        # ── # Optimizing both critics and actors with past transitions from replay buffers ───────────────────────────────────────────────
        if replay.size >= args.learning_starts:
            batch = replay.sample(args.batch_size, device=device)
            metrics = updater.update(batch=batch, isw=None)

        # ── Logging ───────────────────────────────────────────────────────
        if tti % args.log_every == 0:
            msg = f"[TTI {tti}] Buffer={replay.size}"

            if replay.size >= args.learning_starts:
                msg += (
                    f" alpha={metrics['alpha']:.4f}"
                    f" loss_q={metrics['loss_q']:.4f}"
                    f" loss_pi={metrics['loss_pi']:.4f}"
                )
                train_log["tti"].append(int(tti))
                train_log["alpha"].append(float(metrics["alpha"]))
                train_log["loss_q"].append(float(metrics["loss_q"]))
                train_log["loss_pi"].append(float(metrics["loss_pi"]))

            # Eval metrics from MATLAB (always available after finish_tti)
            m = env._last_metrics
            if m:
                msg += (
                    f"\n  MATLAB cell_tput[Mbps]={m.get('total_cell_tput', 0):.2f}"
                    f" jain={m.get('jain_throughput', 0):.3f}"
                    f" pfU={m.get('pf_utility', 0):.2f}"
                    f" inv={m.get('invalid_action_rate', 0):.3f}"
                    f" noop={m.get('no_schedule_rate', 0):.3f}"
                    f" layers/RBG={m.get('avg_layers_per_rbg', 0):.2f}"
                )
                append_matlab_eval(eval_log, tti, m)

            print(msg)

        # ── Periodic checkpoint ───────────────────────────────────────────
        if tti > 0 and tti % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"actor_tti_{tti:06d}.pt")
            torch.save(
                {
                    "actor_state_dict": updater.actor.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "n_rbg": env.n_rbg,
                    "n_layers": env.n_layers,
                    "hidden": args.hidden,
                },
                ckpt_path,
            )
            print(f"[train_matlab] Saved checkpoint → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(args.out_dir, "actor_final.pt")
    torch.save(
        {
            "actor_state_dict": updater.actor.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "n_rbg": env.n_rbg,
            "n_layers": env.n_layers,
            "hidden": args.hidden,
        },
        final_path,
    )
    print(f"[train_matlab] Final actor saved → {final_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    save_logs(
        args.out_dir,
        {"matlab": eval_log},   # wrapped so save_logs works
        train_log,
    )
    plot_training(
        train_log,
        os.path.join(args.out_dir, "training_behavior.png"),
    )
    plot_matlab_eval(
        eval_log,
        os.path.join(args.out_dir, "matlab_eval_metrics.png"),
    )

    env.close()


# ─── CLI args ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="DRL training with MATLAB as environment"
    )
    # TCP
    p.add_argument("--port", type=int, default=5555,
                   help="TCP port to listen on (MATLAB connects here)")
    # Environment config (must match MATLAB's nrDRLScheduler settings)
    p.add_argument("--max_sched_ue", type=int, default=16,
                   help="MaxUEs in MATLAB (eligible UEs per TTI)")
    p.add_argument("--n_layers", type=int, default=16,
                   help="Number of MU-MIMO spatial layers (MATLAB NumLayers)")
    p.add_argument("--n_rbg", type=int, default=18,
                   help="Number of RBGs (must match MATLAB numRBGs)")
    # Training
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ttis", type=int, default=5000,
                   help="Total training TTIs (MATLAB simulation must run ≥ ttis)")
    p.add_argument("--learning_starts", type=int, default=256)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=500)
    # Replay
    p.add_argument("--rb_capacity", type=int, default=72000)
    p.add_argument("--batch_size", type=int, default=64)
    # Fallback action
    p.add_argument("--fallback_action", type=int, default=0)
    # Network
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--n_quantiles", type=int, default=16)
    # DSACD hyperparams
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=1e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)
    # Output
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("outputs", "train_matlab"))

    args = p.parse_args()
    main(args)
