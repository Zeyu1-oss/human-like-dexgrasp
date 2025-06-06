from typing import List, Optional, Union
import torch
from dataclasses import dataclass
from curobo.rollout.cost.cost_base import CostConfig


@dataclass
class JointConsistencyConfig(CostConfig):
    group_allowed_diff: List[float] = None
    selected_joint_groups: List[Union[List[int], List[str]]] = None  # 每组关节名或索引
    group_weight: Optional[List[float]] = None  # 每组惩罚权重

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.group_allowed_diff, list):
            self.group_allowed_diff = self.tensor_args.to_device(self.group_allowed_diff)
        if self.group_weight is not None and isinstance(self.group_weight, list):
            self.group_weight = self.tensor_args.to_device(self.group_weight)
        if self.weight is None:
            self.weight = self.tensor_args.to_device([1.0])


class JointConsistency:
    def __init__(
        self,
        config: JointConsistencyConfig,
        joint_name_to_index_fn: Optional[callable] = None
    ):
        self.tensor_args = getattr(config, "tensor_args", None)
        self.weight = config.weight
        self.group_allowed_diff = config.group_allowed_diff
        self.group_weight = config.group_weight
        self.selected_joint_groups = config.selected_joint_groups

        self.joint_index_to_name = None  

        # joint name 转换为 index
        if joint_name_to_index_fn is not None and isinstance(self.selected_joint_groups[0][0], str):
            
            name_index_pairs = [
                (name, joint_name_to_index_fn(name))
                for group in self.selected_joint_groups for name in group
            ]
            self.joint_index_to_name = {
                idx: name for name, idx in name_index_pairs
            }

            # 替换 selected_joint_groups 为 index 格式
            self.selected_joint_groups = [
                [joint_name_to_index_fn(name) for name in group]
                for group in self.selected_joint_groups
            ]

    def forward(self, joint_state: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        joint_state: [B, H, DOF] - joint angle values across time horizon
        Returns:
            A cost tensor of shape [B, H], representing the joint consistency penalty.
        """
        with torch.autograd.profiler.record_function("cost/joint_consistency"):
            if debug:
                print("=== [JointConsistency Debug Info] ===")
                print("   joint_state shape:", joint_state.shape)
                print("   Mapped joint groups (indices):", self.selected_joint_groups)

            constraint_list = []

            for i, group in enumerate(self.selected_joint_groups):
                group_joints = joint_state[..., group]  # [B, H, len(group)]
                group_var = group_joints.var(dim=-1)    # [B, H]

                threshold = self.group_allowed_diff[i]
                if not isinstance(threshold, torch.Tensor):
                    threshold = torch.tensor(threshold, device=group_var.device, dtype=group_var.dtype)

                violation = torch.clamp(group_var - threshold, min=0.0)  # [B, H]

                weight = self.group_weight[i] if self.group_weight is not None else 1.0
                if not isinstance(weight, torch.Tensor):
                    weight = torch.tensor(weight, device=violation.device, dtype=violation.dtype)

                violation = violation * weight

                if debug:
                    joint_names = None
                    if self.joint_index_to_name is not None:
                        joint_names = [self.joint_index_to_name.get(idx, f"?{idx}") for idx in group]
                    print(f"-- Group {i}: {group}")
                    if joint_names:
                        print(f"   Joint names: {joint_names}")
                    print(f"   Variance (Batch 0): {group_var[0]}")
                    print(f"   Threshold: {threshold}")
                    print(f"   Violation (Batch 0): {violation[0]}")

                constraint_list.append(violation)

            constraint_tensor = torch.stack(constraint_list, dim=-1).sum(dim=-1)  # [B, H]
            final_cost = self.weight * constraint_tensor  # [B, H]

            if debug:
                print(f">> Final Joint Consistency Cost (batch 0): {final_cost[0]}")
                print("Selected joint values (batch0, seed0):", joint_state[0, 0, :].tolist())

            return final_cost