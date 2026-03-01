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
import shutil
from collections import deque
from typing import List

import torch
import numpy as np

from matlab_env_adapter_tuanpa44 import MatlabEnvAdapter
from pprint import pprint
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
    plot_allocation
)

def _ema(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    alpha = float(alpha)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out

def _ema_2d(values_2d: List[List[float]], alpha: float) -> List[List[float]]:
    if not values_2d:
        return []
    alpha = float(alpha)
    n_cols = len(values_2d[0]) if values_2d[0] else 0
    out = [list(values_2d[0])]
    for row in values_2d[1:]:
        prev = out[-1]
        out.append([
            alpha * float(row[c]) + (1.0 - alpha) * float(prev[c])
            for c in range(n_cols)
        ])
    return out

def print_dump_shapes(state: dict):
    import numpy as np
    for k, v in state.items():
        if isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            print(f"{k}: shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"{k}: type={type(v).__name__}")

def _jain_fairness(x: torch.Tensor, eps: float = 1e-12) -> float:
    """Jain's fairness index for non-negative vector x."""
    x = x.float().clamp_min(0.0)
    num = float(x.sum().item()) ** 2
    den = float(x.numel()) * float((x * x).sum().item()) + eps
    return num / den

def ue_rate_under_sinr(eval_env: MatlabEnvAdapter):
    noop = eval_env.noop
    alloc = eval_env._alloc.detach().cpu()
    selected_ues = eval_env.selected_ues
    served = torch.zeros((eval_env.n_ue,), dtype=torch.float32)
    alloc_counts = torch.zeros((eval_env.n_ue,), dtype=torch.float32)
    duration_s = eval_env.tti_ms / 1000.0
    layers_per_rbg_sum = 0.0
    layers_per_rbg_den = 0
    # print(f"Allocation:{alloc}")
    for m in range(eval_env.n_rbg):
        # Get all unique UEs scheduled on this RBG, ignoring NOOPs.
        scheduled_ues_on_rbg = sorted(list(set(u for u in alloc[:, m].tolist() if u != noop)))

        # Find the maximum cross-correlation between any pair of UEs on this RBG
        max_cross_corr_rbg = 0.0
        if len(scheduled_ues_on_rbg) > 1:
            corrs = []
            for i in range(len(scheduled_ues_on_rbg)):
                for j in range(i + 1, len(scheduled_ues_on_rbg)):
                    u1 = scheduled_ues_on_rbg[i]
                    u2 = scheduled_ues_on_rbg[j]
                    corrs.append(eval_env.max_cross_corr[selected_ues[u1], selected_ues[u2], m])
            if corrs:
                max_cross_corr_rbg = max(corrs)

        penalty = 1.0 - max_cross_corr_rbg
        # print(f"Penalty: {penalty}\n")


        scheduled_layers_rbg = len([u for u in alloc[:, m].tolist() if u != noop])
        layers_per_rbg_sum += float(scheduled_layers_rbg)
        layers_per_rbg_den += 1

        for l in range(eval_env.n_layers):
            u = int(alloc[l, m].item())
            if u == noop:
                continue
            global_ue_id = selected_ues[u]
            tbs = eval_env._served_bytes(global_ue_id, m)
            served[global_ue_id] += float(tbs) * penalty
            alloc_counts[global_ue_id] += 1.0
    
    ue_tti = (served * 8.0) / 1e6 / max(duration_s, 1e-9)
    avg_layers_per_rbg = layers_per_rbg_sum / max(layers_per_rbg_den, 1)
    # print(f"alloc_counts: {alloc_counts}")
    # print(f"MCS_Subband: ",eval_env.mcs_subband)
    # print(f"Served: {served}")
    # print(f"UE_Rate: {ue_tti}")
    return ue_tti, avg_layers_per_rbg, alloc_counts

@torch.no_grad()
def evaluate_scheduler_metrics(
    eval_env: MatlabEnvAdapter,
    masks_per_layer: List[torch.Tensor],
    *,
    mode: str,
) -> dict:
    """Compute metrics from current allocation (_alloc) and recorded masks for a single TTI."""
    assert mode in {"sample", "greedy", "random"}, f"mode must be sample/greedy/random, got {mode!r}"

    noop = eval_env.noop
    eps = eval_env.eps

    invalid = 0
    nosched = 0
    decisions = 0

    for l, masks in enumerate(masks_per_layer):
        for m in range(eval_env.n_rbg):
            a = int(eval_env._alloc[l, m].item())
            decisions += 1
            if not bool(masks[m, a].item()):
                invalid += 1
            if a == noop:
                nosched += 1

    ue_tti, avg_layers_per_rbg, alloc_counts = ue_rate_under_sinr(eval_env)
    ue_tti = eval_env.avg_tp
    total_cell_tput = float(ue_tti.sum().item())
    avg_cell_tput = total_cell_tput
    avg_ue_tput = ue_tti

    invalid_action_rate = invalid / max(decisions, 1)
    no_schedule_rate = nosched / max(decisions, 1)
    jain_throughput = _jain_fairness(ue_tti)
    pf_utility = float(torch.log((avg_ue_tput) + eps).sum().item())

    return {
        "mode": mode,
        "eval_ttis": 1,
        "total_cell_tput": float(total_cell_tput),
        "total_ue_tput": ue_tti,
        "avg_cell_tput": float(avg_cell_tput),
        "avg_ue_tput": avg_ue_tput,
        "alloc_counts": alloc_counts,
        "jain_throughput": float(jain_throughput),
        "pf_utility": float(pf_utility),
        "invalid_action_rate": float(invalid_action_rate),
        "no_schedule_rate": float(no_schedule_rate),
        "avg_layers_per_rbg": float(avg_layers_per_rbg),
    }

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

def clear_output_logs_and_plots(out_dir: str):
    """Remove old logs/plots from out_dir before a new training session."""
    os.makedirs(out_dir, exist_ok=True)

    removed_files = 0
    removed_dirs = 0

    # Remove recurrent plot directory.
    alloc_dir = os.path.join(out_dir, "allocation_maps")
    if os.path.isdir(alloc_dir):
        shutil.rmtree(alloc_dir)
        removed_dirs += 1

    # Remove log/plot files while keeping checkpoints (.pt) intact.
    exts_to_clear = {".json", ".log", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    for root, _, files in os.walk(out_dir):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in exts_to_clear:
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed_files += 1
                except OSError:
                    pass

    print(
        f"[train_matlab] Cleared previous logs/plots in {out_dir} "
        f"(files={removed_files}, dirs={removed_dirs})"
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    clear_output_logs_and_plots(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "allocation_maps"), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    verbose = args.verbose

    # ── Environment ───────────────────────────────────────────────────────
    env = MatlabEnvAdapter(
        max_sched_ue=args.max_sched_ue,
        n_layers=args.n_layers,
        n_rbg=args.n_rbg,
        port=args.port,
        verbose=args.verbose,
    )

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print(f"[train_matlab] obs_dim={obs_dim}  act_dim={act_dim}  "
          f"n_rbg={env.n_rbg}  n_layers={env.n_layers}, learning_start{args.learning_starts}")

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
    alloc_eval_log = init_eval_log()

    env.reset()

    # ── Training loop ─────────────────────────────────────────────────────
    for tti in range(args.ttis):
        mode = env.begin_tti()
        if mode == 0:
            break
        assert tti == env.t
        # state = env.dump_state()
        # print_dump_shapes(state)
        # pprint(state,width=200)
        if verbose:
            pprint(env.dump_state(),width=200,sort_dicts=False)

        masks_per_layer: List[torch.Tensor] = []
        for layer_ctx in env.layer_iter():
            masks_per_layer.append(layer_ctx.masks_rbg.detach().cpu())
            # if verbose:
            #     print(f"[LAYER {layer_ctx.layer}]")
            #     print("Current Masks:")
            #     pprint(layer_ctx.masks_rbg)
            #     print("Current Observation:")
            #     pprint(layer_ctx.obs)

            actions_rbg = sample_actions_for_layer(
                actor=updater.actor,
                obs_layer=layer_ctx.obs,
                masks_rbg=layer_ctx.masks_rbg,
                device=device,
                fallback_action=args.fallback_action,
            )

            # if layer_ctx.layer<2:
            #     actions_rbg.fill_(2)
            # else:
            #     actions_rbg.fill_(env.noop)
            # if verbose:
                # print(f"Actions: {actions_rbg}")

            env.apply_layer_actions(layer_ctx, actions_rbg)

            # if verbose:
                # print(f"Allocation: {env._alloc}")

            transitions = env.compute_layer_transitions(layer_ctx)

            for tr in transitions:
                replay.add(tr)

        env.finish_tti()

        # ── # Optimizing both critics and actors with past transitions from replay buffers ───────────────────────────────────────────────
        if replay.size >= args.learning_starts:
            batch = replay.sample(args.batch_size, device=device)
            metrics = updater.update(batch=batch, isw=None)

        # ── Logging ───────────────────────────────────────────────────────
        # Plot allocation map periodically
        if tti > 0 and tti % 10 == 0:
            plot_path = os.path.join(args.out_dir, f"allocation_maps/alloc_tti_{tti:05d}.png")
            plot_allocation(env._alloc, env.n_ue, plot_path)

        if tti % args.log_every == 0:
            msg = f"[TTI {tti}] Buffer={replay.size}"

            alloc_metrics = evaluate_scheduler_metrics(
                env,
                masks_per_layer,
                mode="sample",
            )
            if verbose:
                print(f"Allocation Matrix: {env._alloc}")
                print(f"Final Allocation Counts: {alloc_metrics["alloc_counts"]}")
            append_eval(alloc_eval_log, "sample", tti, alloc_metrics)

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

        # ── Periodic checkpoint ──────────────────────────────────────────
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
        {"matlab": eval_log, "alloc": alloc_eval_log},
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

    # Smooth allocation-derived throughput metrics before plotting
    if alloc_eval_log["sample"]["tti"]:
        smoothed_alloc = dict(alloc_eval_log["sample"])
        smoothed_alloc["avg_cell_tput"] = _ema(
            alloc_eval_log["sample"]["avg_cell_tput"], alpha=0.9
        )
        smoothed_alloc["avg_ue_tput"] = _ema_2d(
            alloc_eval_log["sample"]["avg_ue_tput"], alpha=0.9
        )
    else:
        smoothed_alloc = alloc_eval_log["sample"]

    plot_eval(
        "sample",
        alloc_eval_log["sample"],
        os.path.join(args.out_dir, "python_eval_metrics.png"),
    )

    env.close()


# ─── CLI args ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="DRL training with MATLAB as environment"
    )
    # TCP
    p.add_argument("--port", type=int, default=5556,
                   help="TCP port to listen on (MATLAB connects here)")
    # Environment config (must match MATLAB's nrDRLScheduler settings)
    p.add_argument("--max_sched_ue", type=int, default=4,
                   help="MaxUEs in MATLAB (eligible UEs per TTI)")
    p.add_argument("--n_layers", type=int, default=4,
                   help="Number of MU-MIMO spatial layers (MATLAB NumLayers)")
    p.add_argument("--n_rbg", type=int, default=18,
                   help="Number of RBGs (must match MATLAB numRBGs)")
    # Training
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ttis", type=int, default=500,
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
    p.add_argument("--verbose",type=bool,default=True)

    args = p.parse_args()
    main(args)
