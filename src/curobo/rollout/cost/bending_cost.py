from typing import List, Optional, Union, Callable, Dict
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from curobo.rollout.cost.cost_base import CostConfig

@dataclass
class JointBendingConfig(CostConfig):
    """
    Configuration for joint bending cost with exponential decay.

    Supports optional schedules:
      - weight_schedule: global penalty weight changes over optimization progress
      - target_schedule: per-joint target angles change over progress
    """
    selected_joints:   List[Union[int, str]]   = None
    target_angles:     List[float]             = None
    joint_weights:     Optional[List[float]]   = None
    k:                  float                  = 5.0  # exponential decay rate

    # Optional schedules
    weight_schedule:  Optional[Dict[str, List[float]]]       = None
    target_schedule:  Optional[Dict[str, List[List[float]]]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.tensor_args is not None, "tensor_args must be provided"
        device = self.tensor_args.device
        dtype  = self.tensor_args.dtype

        # Store base (fixed) target angles as a [1,1,J] tensor
        base = torch.tensor(self.target_angles, device=device, dtype=dtype)
        self._base_target = base.view(1,1,-1)

        # Convert joint_weights to tensor [1,1,J] if given
        if self.joint_weights is not None:
            w = torch.tensor(self.joint_weights, device=device, dtype=dtype)
            self.joint_weights = w.view(1,1,-1)

        # Decay rate k as scalar tensor
        self.k = torch.tensor(self.k, device=device, dtype=dtype)

        # Global weight tensor
        if self.weight is None:
            self.weight = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            self.weight = self.weight.to(device=device, dtype=dtype)

        # Prepare NumPy arrays for schedules
        if self.weight_schedule:
            self.ws_prog = np.array(self.weight_schedule["progress"], dtype=np.float32)
            self.ws_val  = np.array(self.weight_schedule["weight"], dtype=np.float32)
        if self.target_schedule:
            self.ts_prog = np.array(self.target_schedule["progress"], dtype=np.float32)
            self.ts_ang  = np.array(self.target_schedule["angles"], dtype=np.float32)

        # Sanity checks
        num_j = self._base_target.numel()
        assert len(self.selected_joints) == num_j, (
            f"selected_joints ({len(self.selected_joints)}) != number of angles ({num_j})"
        )
        if self.joint_weights is not None:
            assert self.joint_weights.numel() == num_j, (
                f"joint_weights ({self.joint_weights.numel()}) != number of angles ({num_j})"
            )

class JointBending(nn.Module):
    """
    Joint bending cost with learnable per-seed scaling factors inferred at runtime.
    """
    def __init__(
        self,
        config: JointBendingConfig,
        mult_range: tuple = (0.8, 1.2),
        joint_name_to_index_fn: Optional[Callable[[str], int]] = None
    ):
        super().__init__()
        self.cfg = config
        self.tensor_args   = config.tensor_args
        self.k             = config.k
        self.weight        = config.weight
        self.joint_weights = config.joint_weights

        # Placeholder for per-seed parameters, to be initialized on first forward
        # per-seed scaling parameters will be registered lazily in forward
        self.m_min, self.m_max = mult_range

        # Map joint names to indices if provided
        self.selected_joints = config.selected_joints
        if joint_name_to_index_fn and isinstance(self.selected_joints[0], str):
            self.selected_joints = [
                joint_name_to_index_fn(name) for name in self.selected_joints
            ]

        # Index tensor for selection
        self.selected_idx = torch.as_tensor(
            self.selected_joints, dtype=torch.long, device=self.tensor_args.device
        )

    def forward(
        self,
        joint_state: torch.Tensor,
        progress: Union[float, torch.Tensor],
        debug: bool = False
    ) -> torch.Tensor:
        # Ensure progress is Python float for interpolation
        if isinstance(progress, torch.Tensor):
            progress = float(progress.cpu().item())

        # 1) Select relevant joints [B, S, J]
        current = joint_state.index_select(dim=-1, index=self.selected_idx)
        B, S, J = current.shape

        # 2) Lazy init of per-seed scaling parameters
        if 'u' not in self._parameters:
            # register a learnable parameter of shape [1, S, 1]
            param = torch.zeros(1, S, 1, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            self.register_parameter('u', nn.Parameter(param))
        # fetch the parameter
        u = self._parameters['u']

        # Compute per-seed mults in [m_min, m_max]
        sig = torch.sigmoid(self.u)        # [1, S, 1]
        mults = self.m_min + (self.m_max - self.m_min) * sig  # [1, S, 1]
        mults = mults.expand(B, S, 1)       # [B, S, 1]

        # 3) Determine base target angles at this progress
        if self.cfg.target_schedule:
            angles = [
                np.interp(progress, self.cfg.ts_prog, self.cfg.ts_ang[:, j])
                for j in range(J)
            ]
            tgt = torch.tensor(
                angles,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            ).view(1, 1, J)
        else:
            tgt = self.cfg._base_target  # [1,1,J]

        # Broadcast target to [B,S,J] then apply per-seed scaling
        tgt = tgt.expand(B, S, J) * mults  # [B,S,J]

        # 4) Under-bend difference
        diff = torch.nn.functional.relu(tgt - current)

        # 5) Determine global weight at this progress
        if self.cfg.weight_schedule:
            w_val = np.interp(progress, self.cfg.ws_prog, self.cfg.ws_val)
            global_w = torch.tensor(
                w_val, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        else:
            global_w = self.weight

        # 6) Exponential decay penalty per joint
        joint_w = self.joint_weights if self.joint_weights is not None else 1.0
        penalty  = diff * joint_w  # [B,S,J]

        # Sum over joints and apply global weight
        cost = penalty.sum(dim=-1)  # [B,S]
        final_cost = cost * global_w * self.weight  # [B,S]

        if debug:
            print("=== JointBending Debug ===")
            print(f"progress={progress:.3f}, global_w={float(global_w):.3f}")
            print("learned mults per seed:", mults[0,:,0].tolist())
            print("k:", float(self.k))
            print("sample cost[0]:", final_cost[0].tolist())

        return final_cost
