from typing import List, Optional, Union, Callable, Dict, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from curobo.rollout.cost.cost_base import CostConfig

@dataclass
class JointBendingConfig(CostConfig):
    """
    Configuration for joint bending cost with optional schedules and random scaling.
    """
    selected_joints:    List[Union[int, str]]               = None
    target_angles:      List[float]                         = None
    joint_weights:      Optional[List[float]]               = None

    # Schedule fields
    weight_schedule:    Optional[Dict[str, List[float]]]    = None
    target_schedule:    Optional[Dict[str, List[List[float]]]] = None

    # Random multiplier range
    mult_range:         Tuple[float, float]                 = (0.1, 1.9)

    def __post_init__(self):
        super().__post_init__()
        assert self.tensor_args is not None, "tensor_args must be provided"
        device, dtype = self.tensor_args.device, self.tensor_args.dtype

        # Base target tensor
        base = torch.tensor(self.target_angles, device=device, dtype=dtype)
        self._base_target = base.view(1, 1, -1)

        # Joint-specific weights
        if self.joint_weights is not None:
            w = torch.tensor(self.joint_weights, device=device, dtype=dtype)
            self.joint_weights = w.view(1, 1, -1)

        # Global weight
        if self.weight is None:
            self.weight = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            self.weight = self.weight.to(device=device, dtype=dtype)

        # Load schedules if provided
        if self.weight_schedule:
            self.ws_prog = np.array(self.weight_schedule["progress"], dtype=np.float32)
            self.ws_val  = np.array(self.weight_schedule["weight"], dtype=np.float32)
        if self.target_schedule:
            self.ts_prog = np.array(self.target_schedule["progress"], dtype=np.float32)
            self.ts_ang  = np.array(self.target_schedule["angles"], dtype=np.float32)

        # Validate selected joints length
        num_j = self._base_target.numel()
        assert len(self.selected_joints) == num_j, \
            f"selected_joints ({len(self.selected_joints)}) != number of angles ({num_j})"
        if self.joint_weights is not None:
            assert self.joint_weights.numel() == num_j

class JointBending(nn.Module):
    """
    Joint bending cost with per-seed random scaling factors initialized on first forward.
    """
    def __init__(
        self,
        config: JointBendingConfig,
        joint_name_to_index_fn: Optional[Callable[[str], int]] = None
    ):
        super().__init__()
        self.cfg = config
        self.tensor_args = config.tensor_args
        self.weight = config.weight
        self.joint_weights = config.joint_weights

        # Min/max for random multipliers from config
        self.m_min, self.m_max = self.cfg.mult_range

        # Placeholder for multipliers, to be initialized on first forward
        self.mults = None

        # Prepare joint indices
        joints = config.selected_joints
        if joint_name_to_index_fn and isinstance(joints[0], str):
            joints = [joint_name_to_index_fn(n) for n in joints]
        device = self.tensor_args.device
        self.selected_idx = torch.tensor(joints, dtype=torch.long, device=device)

    def forward(
        self,
        joint_state: torch.Tensor,
        progress: Union[float, torch.Tensor],
        debug: bool = False
    ) -> torch.Tensor:
        # Ensure progress is float
        if isinstance(progress, torch.Tensor):
            progress = float(progress.cpu().item())

        # Get device and dtype
        device = self.tensor_args.device
        dtype = self.tensor_args.dtype

        current = joint_state.index_select(dim=-1, index=self.selected_idx)
        B, S, J = current.shape

        # Initialize random multipliers on first call
        if self.mults is None:
            rand = torch.rand(B, S, 1, device=device, dtype=dtype)
            self.mults = self.m_min + (self.m_max - self.m_min) * rand

        mults = self.mults

        # Determine target angles schedule or base
        if self.cfg.target_schedule:
            angles = [
                np.interp(progress, self.cfg.ts_prog, self.cfg.ts_ang[:, j])
                for j in range(J)
            ]
            tgt = torch.tensor(
                angles, device=device, dtype=dtype
            ).view(1, 1, J)
        else:
            tgt = self.cfg._base_target.to(device=device, dtype=dtype)

        # Scale by random multipliers
        tgt = tgt.expand(B, S, J) * mults

        # Compute under-bend difference
        diff = torch.relu(tgt - current)

        # Compute global weight schedule or fixed
        if self.cfg.weight_schedule:
            w_val = np.interp(progress, self.cfg.ws_prog, self.cfg.ws_val)
            global_w = torch.tensor(w_val, device=device, dtype=dtype)
        else:
            global_w = self.weight

        # Apply joint and global weights
        joint_w = self.joint_weights if self.joint_weights is not None else 1.0
        penalty = diff * joint_w
        cost = penalty.sum(dim=-1)
        final = cost * global_w * self.weight

        if debug:
            print(f"Initialized mults: {self.mults.flatten().tolist()}")
            print(f"Scaled targets: {tgt[0,0,:min(J,5)].tolist()}")

        return final
