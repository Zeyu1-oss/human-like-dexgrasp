from typing import List, Optional, Union, Callable
import torch
from dataclasses import dataclass
from curobo.rollout.cost.cost_base import CostConfig

@dataclass
class JointConsistencyConfig(CostConfig):
    """
    Configuration for joint consistency cost with optional routing by hand_name.
    """
    # Routing field (may be provided in YAML, ignored internally)
    hand_name: Optional[str] = None
    
    # Allowed variance per joint group
    group_allowed_diff: List[float] = None
    # Groups of joints (as names or indices)
    selected_joint_groups: List[Union[List[int], List[str]]] = None  
    # Optional per-group weights
    group_weight: Optional[List[float]] = None  

    def __post_init__(self):
        # drop routing field
        _ = self.hand_name
        super().__post_init__()
        # bring lists onto tensor device
        if isinstance(self.group_allowed_diff, list):
            self.group_allowed_diff = self.tensor_args.to_device(self.group_allowed_diff)
        if self.group_weight is not None and isinstance(self.group_weight, list):
            self.group_weight = self.tensor_args.to_device(self.group_weight)
        if self.weight is None:
            # default weight
            self.weight = self.tensor_args.to_device([1.0])

class JointConsistency:
    """
    Joint consistency cost: penalizes variance within specified joint groups.
    Supports string names if joint_name_to_index_fn is provided.
    """
    def __init__(
        self,
        config: JointConsistencyConfig,
        joint_name_to_index_fn: Optional[Callable[[str], int]] = None
    ):
        self.tensor_args = getattr(config, "tensor_args", None)
        self.weight = config.weight
        self.group_allowed_diff = config.group_allowed_diff
        self.group_weight = config.group_weight
        self.selected_joint_groups = config.selected_joint_groups
        # routing field ignored
        _ = config.hand_name

        self.joint_index_to_name = None
        # convert group names to indices if needed
        if joint_name_to_index_fn is not None and isinstance(self.selected_joint_groups[0][0], str):
            # build name->index mapping
            name_index_pairs = []
            for group in self.selected_joint_groups:
                for name in group:
                    idx = joint_name_to_index_fn(name)
                    name_index_pairs.append((name, idx))
            self.joint_index_to_name = {idx: name for name, idx in name_index_pairs}
            # rewrite groups as index lists
            self.selected_joint_groups = [
                [joint_name_to_index_fn(name) for name in group]
                for group in self.selected_joint_groups
            ]

    def forward(self, joint_state: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        joint_state: [B, H, DOF]
        returns cost tensor [B, H]
        """
        with torch.autograd.profiler.record_function("cost/joint_consistency"):
            if debug:
                print("=== [JointConsistency Debug] state shape:", joint_state.shape)
                print(" groups:", self.selected_joint_groups)

            violations = []
            for i, group in enumerate(self.selected_joint_groups):
                vals = joint_state[..., group]           # [B, H, G]
                var = vals.var(dim=-1)                  # [B, H]
                thr = self.group_allowed_diff[i]
                if not isinstance(thr, torch.Tensor):
                    thr = torch.tensor(thr, device=var.device, dtype=var.dtype)
                vio = torch.clamp(var - thr, min=0.0)
                w = self.group_weight[i] if self.group_weight is not None else 1.0
                if not isinstance(w, torch.Tensor):
                    w = torch.tensor(w, device=vio.device, dtype=vio.dtype)
                vio = vio * w
                violations.append(vio)
                if debug:
                    names = None
                    if self.joint_index_to_name:
                        names = [self.joint_index_to_name.get(idx, f"?{idx}") for idx in group]
                        print(f" group {i} names {names}")
                        print(f"  var[0]={var[0]}")
                        print(f"  vio[0]={vio[0]}")

            total = torch.stack(violations, dim=-1).sum(dim=-1)  # [B, H]
            cost = self.weight * total
            if debug:
                print(" final cost[0]=", cost[0])
            return cost