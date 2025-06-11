from typing import List, Optional, Union, Callable, Dict
from dataclasses import dataclass
import torch
import numpy as np
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

class JointBending:

    def __init__(
        self,
        config: JointBendingConfig,
        joint_name_to_index_fn: Optional[Callable[[str], int]] = None
    ):
        self.cfg = config
        self.tensor_args   = config.tensor_args
        self.k             = config.k
        self.weight        = config.weight
        self.joint_weights = config.joint_weights

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
        """
        Compute bending cost.

        Args:
          joint_state: Tensor of shape [B, S, DOF]
          progress:    float or 0-dim tensor, optimization progress in [0,1]
          debug:       if True, print debug info

        Returns:
          Tensor of shape [B, S]: bending cost per time step
        """
        # Ensure progress is a Python float for np.interp
        if isinstance(progress, torch.Tensor):
            progress = float(progress.cpu().item())

        # 1) Select relevant joints [B, S, J]
        current = joint_state.index_select(dim=-1, index=self.selected_idx)

        # 2) Determine target angles at this progress
        if self.cfg.target_schedule:
            angles = [
                np.interp(progress, self.cfg.ts_prog, self.cfg.ts_ang[:, j])
                for j in range(self.cfg.ts_ang.shape[1])
            ]  # list of J floats
            tgt = torch.tensor(
                angles,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            ).view(1, 1, -1)
        else:
            tgt = self.cfg._base_target

        # 3) Under-bend difference
        diff = torch.nn.functional.relu(tgt - current)

        # 4) Determine global weight at this progress
        if self.cfg.weight_schedule:
            w_val = np.interp(progress, self.cfg.ws_prog, self.cfg.ws_val)
            global_w = torch.tensor(
                w_val,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
        else:
            global_w = self.weight

        # 5) Exponential decay penalty per joint
        joint_w = self.joint_weights if self.joint_weights is not None else 1.0
        exp_term = torch.exp(-self.k * diff)
        penalty  = diff * joint_w

        # Sum over joints and apply global weight
        cost = penalty.sum(dim=-1)
        final_cost = cost * global_w * self.weight
        debug= True
        if debug:
            print("=== JointBending Debug ===")
            print(f"progress={progress:.3f}, global_w={float(global_w):.3f}")
            print("target angles:", tgt.flatten().tolist())
            print("k:", float(self.k))
            print("sample cost:", final_cost[0, :5].tolist())

        return final_cost