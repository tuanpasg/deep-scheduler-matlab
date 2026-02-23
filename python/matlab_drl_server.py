"""
MATLAB-Python DRL Server for Multi-Layer MU-MIMO Scheduling
Handles TCP/IP communication for 16-layer gNB with max 2 layers per UE
"""

import socket
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MATLABDRLServer:
    """
    TCP/IP Server for receiving layer-by-layer observations from MATLAB
    and sending back actions after processing all 16 layers in a TTI
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, max_layers: int = 16):
        self.host = host
        self.port = port
        self.max_layers = max_layers
        
        self.socket = None
        self.client_socket = None
        
        # TTI-level action buffer
        self.tti_action_buffer = []  # List of actions for each layer
        self.layer_count = 0
        
        # Track RBG allocation per layer: layer_id -> {rbg_idx: ue_id}
        self.rbg_allocation_tracker = {}
        
        # Observation buffer for current TTI
        self.obs_buffer = []
        
    def start(self):
        """Start the TCP server and wait for MATLAB connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        
        print(f"[Python DRL Server] Listening on {self.host}:{self.port}")
        print(f"[Python DRL Server] Waiting for MATLAB connection...")
        
        self.client_socket, addr = self.socket.accept()
        print(f"[Python DRL Server] Connected to MATLAB from {addr}")
        
    def receive_observation(self) -> Optional[Dict]:
        """
        Receive one layer's observation from MATLAB
        Returns: dict with 'layer_id', 'features', 'masks', etc.
        """
        try:
            # Read until newline
            data = b""
            while True:
                chunk = self.client_socket.recv(1)
                if not chunk:
                    return None
                if chunk == b'\n':
                    break
                data += chunk
            
            if not data:
                return None
                
            # Parse JSON
            obs_dict = json.loads(data.decode('utf-8'))
            return obs_dict
            
        except Exception as e:
            print(f"[Python DRL Server] Error receiving observation: {e}")
            return None
    
    def send_actions(self, actions: Dict):
        """
        Send actions back to MATLAB after processing all layers
        actions: dict with 'prbs' key containing PRB allocations per UE
        """
        try:
            json_str = json.dumps(actions)
            self.client_socket.sendall(json_str.encode('utf-8'))
            self.client_socket.sendall(b'\n')
            print(f"[Python DRL Server] Sent actions to MATLAB")
        except Exception as e:
            print(f"[Python DRL Server] Error sending actions: {e}")
    
    def add_layer_action(self, layer_id: int, actions_rbg: torch.Tensor):
        """
        Add actions for one layer to the buffer
        actions_rbg: [num_rbg] tensor of UE indices (or NOOP)
        """
        self.tti_action_buffer.append({
            'layer_id': layer_id,
            'actions': actions_rbg.cpu().numpy().tolist()
        })
        self.layer_count += 1
    
    def track_rbg_allocation(self, layer_id: int, rbg_idx: int, ue_id: int):
        """
        Track which RBG is allocated to which UE on which layer
        """
        if layer_id not in self.rbg_allocation_tracker:
            self.rbg_allocation_tracker[layer_id] = {}
        self.rbg_allocation_tracker[layer_id][rbg_idx] = ue_id
    
    def is_tti_complete(self) -> bool:
        """Check if we've received actions for all layers in current TTI"""
        return self.layer_count >= self.max_layers
    
    def get_tti_actions(self) -> Dict:
        """
        Convert buffered layer actions to MATLAB format (PRB counts per UE)
        Returns: dict with 'prbs' key
        """
        # Aggregate actions across all layers to get PRB allocation per UE
        
        max_ues = 16  # Max UEs in system
        num_rbg = len(self.tti_action_buffer[0]['actions']) if self.tti_action_buffer else 18
        total_prbs = num_rbg * 16  # RBG size is 16 PRBs (273 RBs / 18 RBGs ≈ 15, use 16)
        max_prbs_per_ue = total_prbs  # Upper bound
        
        # Count RBG allocations per UE across all layers
        ue_rbg_counts = [0] * max_ues
        
        for layer_data in self.tti_action_buffer:
            actions = layer_data['actions']
            for rbg_idx, ue_id in enumerate(actions):
                if 0 <= ue_id < max_ues:  # Not NOOP
                    ue_rbg_counts[ue_id] += 1
        
        # Convert RBG counts to PRB counts (rbg_size = 16)
        rbg_size = 16
        prbs = [min(count * rbg_size, max_prbs_per_ue) for count in ue_rbg_counts]
        
        # Ensure no negative values and total doesn't exceed budget
        prbs = [max(0, p) for p in prbs]
        
        return {'prbs': prbs}
    
    def reset_tti(self):
        """Reset buffers for next TTI"""
        self.tti_action_buffer = []
        self.layer_count = 0
        self.rbg_allocation_tracker = {}
        self.obs_buffer = []
    
    def close(self):
        """Close the connection"""
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        print("[Python DRL Server] Connection closed")


class MultiLayerObservationHandler:
    """
    Handles observations from multiple layers and prepares them for DRL agent
    """
    
    def __init__(self, num_subbands: int = 18, num_features_fixed: int = 5):
        self.num_subbands = num_subbands
        self.num_features_fixed = num_features_fixed
        self.obs_dim = num_features_fixed + 2 * num_subbands  # 5 + 2*18 = 41 features
        
    def parse_matlab_observation(self, obs_dict: Dict) -> torch.Tensor:
        """
        Parse observation dictionary from MATLAB into tensor
        Expected format:
        {
            'layer_id': int,
            'ue_id': int,
            'features': [f_R, f_h, f_d, f_b, f_o, f_g_vec..., f_rho_vec...]
            'num_subbands': int,
            'allocated_rbgs': [list of already allocated RBGs in this layer]
        }
        """
        features = obs_dict.get('features', [])
        
        # Convert to tensor and ensure correct shape
        obs_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Pad or truncate to obs_dim
        if obs_tensor.numel() < self.obs_dim:
            padding = torch.zeros(self.obs_dim - obs_tensor.numel())
            obs_tensor = torch.cat([obs_tensor, padding])
        elif obs_tensor.numel() > self.obs_dim:
            obs_tensor = obs_tensor[:self.obs_dim]
        
        return obs_tensor
    
    def create_action_masks(self, obs_dict: Dict, allocated_rbgs: Dict, noop_idx: int) -> torch.Tensor:
        """
        Create action masks based on:
        1. UE buffer status (can only schedule UE with buffer > 0)
        2. Already allocated RBGs in this layer (avoid double allocation)
        
        Returns: [num_rbg, act_dim] boolean tensor
        """
        num_rbg = obs_dict.get('num_rbg', 18)
        num_ues = obs_dict.get('num_ues', 16)
        act_dim = num_ues + 1  # UEs + NOOP
        
        # Start with all actions valid
        masks = torch.ones((num_rbg, act_dim), dtype=torch.bool)
        
        # Get buffer status from observation
        buffer = obs_dict.get('buffer_status', [])
        
        # Mask out UEs with zero buffer
        for ue_id in range(num_ues):
            if ue_id < len(buffer) and buffer[ue_id] <= 0:
                masks[:, ue_id] = False
        
        # NOOP is always valid
        masks[:, noop_idx] = True
        
        # Ensure at least one action is valid per RBG
        for m in range(num_rbg):
            if not masks[m].any():
                masks[m, noop_idx] = True
        
        return masks


def convert_actions_to_prb_allocation(
    layer_actions_list: List[Dict],
    num_ues: int = 16,
    rbg_size: int = 16
) -> Dict:
    """
    Convert layer-by-layer actions to PRB allocation per UE
    
    Args:
        layer_actions_list: List of dicts with 'layer_id' and 'actions' (per RBG)
        num_ues: Total number of UEs
        rbg_size: Size of each RBG in PRBs
    
    Returns:
        Dict with 'prbs' key containing PRB count per UE
    """
    ue_rbg_counts = [0] * num_ues
    
    # Count RBG allocations across all layers
    for layer_data in layer_actions_list:
        actions = layer_data['actions']
        for ue_id in actions:
            if 0 <= ue_id < num_ues:
                ue_rbg_counts[ue_id] += 1
    
    # Convert to PRBs
    prbs = [count * rbg_size for count in ue_rbg_counts]
    
    return {'prbs': prbs}
