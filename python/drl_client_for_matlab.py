import argparse
import json
import socket
import numpy as np
import torch

# ---- you already have these in your project ----
# from DSACD_multibranch import MultiBranchActor, ensure_nonempty_mask, apply_action_mask_to_logits
# from your_obs_builder import build_observation_from_matlab, build_action_masks_from_matlab

NL = b"\n"

def recv_json_line(sock: socket.socket, rxbuf: bytearray) -> dict:
    while True:
        i = rxbuf.find(NL)
        if i != -1:
            line = rxbuf[:i]
            del rxbuf[:i+1]
            if not line:
                continue
            return json.loads(line.decode("utf-8"))
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("Socket closed by MATLAB")
        rxbuf.extend(chunk)

def send_json_line(sock: socket.socket, obj: dict):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)

@torch.no_grad()
def sample_actions_for_layer(actor, obs_layer: torch.Tensor, masks_rbg: torch.Tensor,
                             device: torch.device, fallback_action: int) -> torch.Tensor:
    # obs_layer: [obs_dim]
    obs_b = obs_layer.unsqueeze(0).to(device)          # [1, obs_dim]
    logits_all = actor.forward_all(obs_b).squeeze(0)   # [NRBG, A]

    # ensure_nonempty_mask + apply_action_mask_to_logits are from your code
    masks_rbg = ensure_nonempty_mask(masks_rbg.to(device), fallback_action=fallback_action)
    logits_all = apply_action_mask_to_logits(logits_all, masks_rbg)

    dist = torch.distributions.Categorical(logits=logits_all)  # batched over NRBG
    actions = dist.sample()                                    # [NRBG]
    return actions.cpu()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # TODO: load your trained actor here
    actor = ...  # MultiBranchActor(...)
    actor.eval().to(device)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    rxbuf = bytearray()
    print(f"[Python] Connected to MATLAB server {args.host}:{args.port}")

    while True:
        msg = recv_json_line(sock, rxbuf)
        if msg.get("type") != "TTI_OBS":
            continue

        max_layers = int(msg["max_layers"])           # 16
        num_rbg = int(msg["num_rbg"])                 # NRBG
        max_ues = int(msg["max_ues"])                 # 16
        eligible = msg["eligible_ues"]                # list of RNTI (1..MaxUEs)
        features = np.array(msg["features"], dtype=np.float32)  # [MaxUEs, featDim]

        # ---- Python keeps dynamic allocation state within this TTI ----
        layers_allocated = torch.zeros(max_ues, dtype=torch.long)    # per UE
        layers_on_rbg = torch.zeros(num_rbg, dtype=torch.long)       # per RBG
        scheduled_ues_on_rbg = [set() for _ in range(num_rbg)]       # UE set per RBG

        actions_buffer = []

        for layer_idx in range(max_layers):
            # Build obs_layer + masks from your own builders
            obs_layer = build_observation_from_matlab(
                matlab_features=features,
                eligible_ues=eligible,
                layers_allocated=layers_allocated,
                layers_on_rbg=layers_on_rbg,
                scheduled_ues_on_rbg=scheduled_ues_on_rbg,
                layer_idx=layer_idx,
                num_rbg=num_rbg,
                max_ues=max_ues
            )
            masks_rbg = build_action_masks_from_matlab(
                matlab_features=features,
                eligible_ues=eligible,
                layers_allocated=layers_allocated,
                layers_on_rbg=layers_on_rbg,
                scheduled_ues_on_rbg=scheduled_ues_on_rbg,
                num_rbg=num_rbg,
                max_ues=max_ues,
                max_layers_per_ue=2,
                max_layers_per_rbg=16
            )

            # actions_rbg: [NRBG], each entry in {0..max_ues} where max_ues means NO_ALLOC (or your choice)
            actions_rbg = sample_actions_for_layer(
                actor=actor,
                obs_layer=obs_layer,
                masks_rbg=masks_rbg,
                device=device,
                fallback_action=max_ues  # if you define NO_ALLOC as max_ues
            )

            actions_buffer.append(actions_rbg)

            # Update allocation state (convert action to UE id or NO_ALLOC)
            for m in range(num_rbg):
                a = int(actions_rbg[m].item())
                if a == max_ues:
                    continue  # NO_ALLOC
                ue = a  # if your action is UE index directly 0..max_ues-1 then map carefully here
                # ---- enforce constraints ----
                if ue < 0 or ue >= max_ues:
                    continue
                if (ue + 1) not in eligible:          # if eligible uses RNTI 1-based
                    continue
                if layers_allocated[ue] >= 2:
                    continue
                if layers_on_rbg[m] >= 16:
                    continue
                if ue in scheduled_ues_on_rbg[m]:
                    continue

                layers_allocated[ue] += 1
                layers_on_rbg[m] += 1
                scheduled_ues_on_rbg[m].add(ue)

        # Stack to allocationMatrix [NRBG, 16]
        actions_layers = torch.stack(actions_buffer, dim=0)     # [16, NRBG]
        allocation = actions_layers.transpose(0, 1).tolist()    # [NRBG, 16]

        send_json_line(sock, {"type": "TTI_ALLOC", "allocation": allocation})

if __name__ == "__main__":
    main()
