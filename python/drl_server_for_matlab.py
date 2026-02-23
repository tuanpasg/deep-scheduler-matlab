"""
DRL Server for MATLAB SchedulerDRL
===================================

This server receives observations from MATLAB's SchedulerDRL.m and
sends back PRB allocation decisions made by the trained DRL agent.

Protocol (matching SchedulerDRL.m):
  MATLAB -> Python: Layer observations (JSON, one per layer)
  Python -> MATLAB: PRB counts per UE (JSON array)

The server accumulates observations for all 16 layers, makes scheduling
decisions, and returns the PRB allocation.
"""

import socket
import json
import argparse
import torch
import numpy as np

from DSACD_multibranch import MultiBranchActor


class MATLABSchedulerServer:
    """
    TCP server that listens for MATLAB SchedulerDRL connections.
    """
    
    def __init__(self, port=5555, actor_path=None, device='cpu', verbose=True):
        """
        Args:
            port: Port to listen on (default 5555, matches SchedulerDRL.m)
            actor_path: Path to saved actor model weights
            device: Computation device
            verbose: Print debug messages
        """
        self.port = port
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Load actor network if provided
        self.actor = None
        if actor_path:
            self.load_actor(actor_path)
        
        self.server_socket = None
        self.client_socket = None
        
    def load_actor(self, path):
        """Load trained actor network."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Determine network configuration from checkpoint
            obs_dim = checkpoint.get('obs_dim', 617)
            n_rbg = checkpoint.get('n_rbg', 18)
            act_dim = checkpoint.get('act_dim', 11)
            hidden = checkpoint.get('hidden', 32)
            
            self.actor = MultiBranchActor(
                obs_dim=obs_dim,
                n_rbg=n_rbg,
                act_dim=act_dim,
                hidden=hidden
            ).to(self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.eval()
            
            print(f"[Server] Loaded actor from {path}")
            print(f"[Server] Config: obs_dim={obs_dim}, n_rbg={n_rbg}, act_dim={act_dim}")
            
        except Exception as e:
            print(f"[Server] Failed to load actor: {e}")
            print(f"[Server] Will use random policy")
            self.actor = None
    
    def start(self):
        """Start the TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        
        print(f"[Server] DRL Server listening on port {self.port}")
        print(f"[Server] Waiting for MATLAB SchedulerDRL to connect...")
        print(f"[Server] Run MU_MIMO.m in MATLAB now")
        print(f"[Server] Press Ctrl+C to stop\n")
        
        while True:
            try:
                self.client_socket, addr = self.server_socket.accept()
                print(f"[Server] MATLAB connected from {addr}")
                
                self.handle_client()
                
            except KeyboardInterrupt:
                print("\n[Server] Shutting down...")
                break
            except Exception as e:
                print(f"[Server] Error: {e}")
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
        
        self.cleanup()
    
    def handle_client(self):
        """Handle communication with MATLAB client."""
        buffer = b""
        tti_count = 0
        layer_count = 0
        
        while True:
            try:
                # Receive data from MATLAB
                chunk = self.client_socket.recv(4096)
                if not chunk:
                    print("[Server] MATLAB disconnected")
                    break
                
                buffer += chunk
                
                # Process complete JSON messages (newline-delimited)
                while b'\n' in buffer:
                    message, buffer = buffer.split(b'\n', 1)
                    
                    if not message.strip():
                        continue
                    
                    # Parse JSON
                    try:
                        data = json.loads(message.decode('utf-8'))
                    except json.JSONDecodeError as e:
                        print(f"[Server] JSON decode error: {e}")
                        continue
                    
                    # Check message type
                    if data.get('type') == 'LAYER_OBSERVATION':
                        # New protocol: receive observation for one layer with ALL UE features
                        layer_id = data.get('layer_id', 0)
                        num_ues = data.get('num_ues', 16)
                        num_rbg = data.get('num_rbg', 18)
                        features = data.get('features', [])
                        eligible_ues = data.get('eligible_ues', [])
                        buffer_status = data.get('buffer_status', [0] * num_ues)
                        
                        # Allocation state
                        layers_allocated = data.get('layers_allocated_per_ue', [0] * num_ues)
                        layers_per_rbg = data.get('layers_used_per_rbg', [0] * num_rbg)
                        
                        if layer_id == 0:
                            tti_count += 1
                            layer_count = 0
                            if self.verbose:
                                print(f"\n[Server] === TTI {tti_count} ===")
                        
                        layer_count += 1
                        
                        if self.verbose:
                            if layer_id < 2 or layer_id >= 14:
                                print(f"[Server] Layer {layer_id:2d}: Received obs with {num_ues} UE features, {num_rbg} RBGs")
                                print(f"          Eligible UEs: {eligible_ues}")
                            elif layer_id == 2:
                                print(f"[Server] ... (layers 2-13) ...")
                        
                        # Generate actions for this layer
                        # actions[rbg] = UE_ID (1-indexed) or 0 (NO_ALLOC)
                        actions = self.generate_layer_actions(
                            num_rbg=num_rbg,
                            num_ues=num_ues,
                            eligible_ues=eligible_ues,
                            buffer_status=buffer_status,
                            layers_allocated=layers_allocated,
                            layers_per_rbg=layers_per_rbg,
                            features=features
                        )
                        
                        # Send actions back to MATLAB
                        self.send_layer_actions(actions)
                        
                        if self.verbose and (layer_id < 2 or layer_id >= 14):
                            num_allocated = sum(1 for a in actions if a > 0)
                            print(f"[Server] Layer {layer_id:2d}: Sent {num_allocated}/{num_rbg} allocations")
                    
                    elif data.get('type') == 'TTI_OBS':
                        # Single message per TTI: Python loops over all layers and returns full allocation matrix
                        tti_count += 1
                        
                        max_ues = data.get('max_ues', 16)
                        max_layers = data.get('max_layers', 16)
                        max_layers_per_ue = data.get('max_layers_per_ue', 2)
                        num_rbg = data.get('num_rbg', 18)
                        eligible_ues = data.get('eligible_ues', [])
                        features = data.get('features', [])  # [max_ues, feat_dim]
                        
                        if self.verbose:
                            print(f"\n[Server] === TTI {tti_count} (Frame {data.get('frame', 0)}, Slot {data.get('slot', 0)}) ===")
                            print(f"[Server] Received: {max_ues} UEs, {max_layers} layers, {num_rbg} RBGs")
                            print(f"[Server] Eligible UEs: {eligible_ues}")
                        
                        # Generate full allocation matrix: [num_rbg, max_layers]
                        # allocation[rbg][layer] = UE_ID (1-indexed) or 0 (NO_ALLOC)
                        allocation_matrix = self.generate_tti_allocation(
                            num_rbg=num_rbg,
                            max_layers=max_layers,
                            max_ues=max_ues,
                            max_layers_per_ue=max_layers_per_ue,
                            eligible_ues=eligible_ues,
                            features=features
                        )
                        
                        # Send allocation matrix back to MATLAB
                        self.send_tti_allocation(allocation_matrix)
                        
                        if self.verbose:
                            total_allocs = sum(1 for rbg in allocation_matrix for ue in rbg if ue > 0)
                            print(f"[Server] Sent allocation matrix: {total_allocs} total allocations across {num_rbg}x{max_layers} slots")
                    
            except ConnectionResetError:
                print("[Server] Connection reset by MATLAB")
                break
            except Exception as e:
                print(f"[Server] Error in handle_client: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
    
    def generate_layer_actions(self, num_rbg, num_ues, eligible_ues, buffer_status, 
                               layers_allocated, layers_per_rbg, features):
        """
        Generate actions for one layer.
        
        Args:
            num_rbg: Number of RBGs
            num_ues: Total number of UEs
            eligible_ues: List of eligible UE IDs (1-indexed)
            buffer_status: Buffer occupancy per UE
            layers_allocated: How many layers each UE already has
            layers_per_rbg: How many layers each RBG has used
            features: UE features matrix [num_ues x feat_dim]
            
        Returns:
            List of actions [num_rbg], where action = UE_ID (1-indexed) or 0 (NO_ALLOC)
        """
        actions = []
        
        if self.actor is None:
            # Random policy with constraints
            for rbg in range(num_rbg):
                # Find valid UEs for this RBG
                valid_ues = []
                for ue in eligible_ues:
                    if ue > 0 and ue <= num_ues:
                        # Check constraints
                        if buffer_status[ue - 1] > 0:  # Has data
                            if layers_allocated[ue - 1] < 2:  # UE layer limit
                                if layers_per_rbg[rbg] < 16:  # RBG layer limit
                                    valid_ues.append(ue)
                
                # Select action
                if len(valid_ues) > 0 and np.random.rand() > 0.3:  # 70% allocate
                    action = int(np.random.choice(valid_ues))
                else:
                    action = 0  # NO_ALLOC
                
                actions.append(action)
        else:
            # Use trained actor (placeholder - would need full implementation)
            for rbg in range(num_rbg):
                # Simple allocation for now
                if len(eligible_ues) > 0 and rbg % 2 == 0:
                    action = int(eligible_ues[rbg % len(eligible_ues)])
                else:
                    action = 0
                actions.append(action)
        
        return actions
    
    def send_layer_actions(self, actions):
        """Send layer actions back to MATLAB."""
        response = {'actions': actions}
        json_str = json.dumps(response) + '\n'
        self.client_socket.sendall(json_str.encode('utf-8'))
    
    def generate_prb_allocation(self, num_ues=16):
        """
        Generate PRB allocation for all UEs.
        
        Args:
            num_ues: Number of UEs
            
        Returns:
            List of PRB counts per UE
        """
        if self.actor is None:
            # Random allocation if no trained policy
            total_prbs = 273
            prbs = np.random.dirichlet(np.ones(num_ues)) * total_prbs
            prbs = np.floor(prbs).astype(int)
            # Distribute remaining PRBs
            remaining = total_prbs - prbs.sum()
            prbs[:remaining] += 1
        else:
            # Use trained actor (would need full observation)
            # For now, placeholder
            prbs = np.zeros(num_ues, dtype=int)
            prbs[:8] = 30  # Simple allocation
        
        return prbs.tolist()
    
    def send_prb_allocation(self, prbs):
        """Send PRB allocation back to MATLAB."""
        response = {'prbs': prbs}
        json_str = json.dumps(response) + '\n'
        self.client_socket.sendall(json_str.encode('utf-8'))
        
        if self.verbose:
            print(f"[Server] Sent PRB allocation: {prbs[:8]}... (first 8 UEs)")
    
    def generate_tti_allocation(self, num_rbg, max_layers, max_ues, max_layers_per_ue, eligible_ues, features):
        """
        Generate full allocation matrix for one TTI.
        Python loops through all layers and generates [num_rbg, max_layers] matrix.
        
        Args:
            num_rbg: Number of RBGs (e.g., 18)
            max_layers: Maximum spatial layers (e.g., 16)
            max_ues: Total UEs (e.g., 16)
            max_layers_per_ue: Max layers per UE (e.g., 2)
            eligible_ues: List of eligible UE IDs (1-indexed)
            features: UE features [max_ues x feat_dim]
            
        Returns:
            allocation_matrix: [num_rbg][max_layers] where value = UE_ID (1-indexed) or 0
        """
        # Initialize allocation matrix
        allocation_matrix = [[0 for _ in range(max_layers)] for _ in range(num_rbg)]
        
        # Track constraints
        layers_allocated_per_ue = [0] * max_ues  # How many layers each UE has
        layers_used_per_rbg = [0] * num_rbg  # How many layers each RBG has
        
        # Extract buffer status from features if available
        # Assuming features format: [tput, rank, alloc, buffer, wbcqi, ...]
        buffer_status = []
        if len(features) >= max_ues and len(features[0]) >= 4:
            buffer_status = [features[ue][3] for ue in range(max_ues)]
        else:
            buffer_status = [1e6] * max_ues  # Assume all have data
        
        if self.actor is None:
            # Random policy with constraints
            for layer_idx in range(max_layers):
                for rbg in range(num_rbg):
                    # Find valid UEs for this RBG on this layer
                    valid_ues = []
                    for ue_id in eligible_ues:
                        if ue_id > 0 and ue_id <= max_ues:
                            ue_idx = ue_id - 1
                            # Check constraints
                            if buffer_status[ue_idx] > 0:  # Has data
                                if layers_allocated_per_ue[ue_idx] < max_layers_per_ue:  # UE layer limit
                                    if layers_used_per_rbg[rbg] < max_layers:  # RBG layer limit
                                        valid_ues.append(ue_id)
                    
                    # Select action with 60% allocation probability
                    if len(valid_ues) > 0 and np.random.rand() > 0.4:
                        ue_id = int(np.random.choice(valid_ues))
                        allocation_matrix[rbg][layer_idx] = ue_id
                        
                        # Update constraints
                        layers_allocated_per_ue[ue_id - 1] += 1
                        layers_used_per_rbg[rbg] += 1
                    else:
                        allocation_matrix[rbg][layer_idx] = 0  # NO_ALLOC
        else:
            # Use trained actor (placeholder - would need full DRL implementation)
            for layer_idx in range(max_layers):
                for rbg in range(num_rbg):
                    # Simple round-robin allocation for testing
                    if len(eligible_ues) > 0:
                        ue_id = eligible_ues[(rbg + layer_idx) % len(eligible_ues)]
                        if layers_allocated_per_ue[ue_id - 1] < max_layers_per_ue:
                            allocation_matrix[rbg][layer_idx] = ue_id
                            layers_allocated_per_ue[ue_id - 1] += 1
        
        return allocation_matrix
    
    def send_tti_allocation(self, allocation_matrix):
        """Send full allocation matrix back to MATLAB."""
        response = {
            'type': 'TTI_ALLOC',
            'allocation': allocation_matrix
        }
        json_str = json.dumps(response) + '\n'
        self.client_socket.sendall(json_str.encode('utf-8'))
    
    def cleanup(self):
        """Clean up sockets."""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[Server] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='DRL Server for MATLAB SchedulerDRL')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port to listen on (default: 5555)')
    parser.add_argument('--actor', type=str, default=None,
                       help='Path to saved actor model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print debug messages')
    
    args = parser.parse_args()
    
    server = MATLABSchedulerServer(
        port=args.port,
        actor_path=args.actor,
        device=args.device,
        verbose=args.verbose
    )
    
    server.start()


if __name__ == '__main__':
    main()
