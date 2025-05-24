from typing import List, Optional, Union
import torch
from dataclasses import dataclass
from curobo.rollout.cost.cost_base import CostConfig

@dataclass
class JointBendingConfig(CostConfig):
    """
    Configuration for joint bending cost:
      selected_joints:  List of joint indices or names to apply bending incentive
      target_angles:    Desired joint angles in radians
      joint_weights:    Optional per-joint weighting factors
    """
    selected_joints: List[Union[int, str]] = None
    target_angles:    List[float]           = None
    joint_weights:    Optional[List[float]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.tensor_args is not None, "tensor_args must be provided"

        # Convert target_angles to tensor and reshape for broadcasting: [1, 1, J]
        target = torch.tensor(
            self.target_angles,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        self.target_angles = target.view(1, 1, -1)

        # Convert joint_weights if provided and reshape: [1, 1, J]
        if self.joint_weights is not None:
            w = torch.tensor(
                self.joint_weights,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
            self.joint_weights = w.view(1, 1, -1)

        # Initialize global weight as a scalar tensor
        if self.weight is None:
            self.weight = torch.tensor(
                1.0,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
        else:
            self.weight = self.weight.to(
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )

        # Check consistency of lengths
        num_j = self.target_angles.numel()
        assert len(self.selected_joints) == num_j, (
            f"selected_joints length ({len(self.selected_joints)}) must match"
            f" number of target_angles ({num_j})"
        )
        if self.joint_weights is not None:
            assert self.joint_weights.numel() == num_j, (
                f"joint_weights length ({self.joint_weights.numel()}) must match"
                f" number of selected_joints ({len(self.selected_joints)})"
            )

class JointBending:
    """
    Cost that encourages specified joints to move toward target angles.
    Input:
      joint_state: Tensor of shape [B, S, DOF] containing joint angles per seed.
    Output:
      final_cost: Tensor of shape [B, S], bending cost per seed.

    Loss is:
        loss[b, s] = weight * sum_j [ joint_weight_j * (theta_{b,s,j} - target_j)**2 ]
    """
    def __init__(
        self,
        config: JointBendingConfig,
        joint_name_to_index_fn: Optional[callable] = None
    ):
        self.tensor_args         = getattr(config, "tensor_args", None)
        self.weight              = config.weight        # scalar tensor
        self.target_angles       = config.target_angles # shape [1,1,J]
        self.joint_weights       = config.joint_weights # shape [1,1,J] or None
        self.selected_joints     = config.selected_joints
        self.joint_index_to_name = None

        # Map names to indices if needed
        if joint_name_to_index_fn is not None and isinstance(self.selected_joints[0], str):
            mapped = []
            self.joint_index_to_name = {}
            for name in self.selected_joints:
                idx = joint_name_to_index_fn(name)
                mapped.append(idx)
                self.joint_index_to_name[idx] = name
            self.selected_joints = mapped

        # Cache the index tensor for faster forward calls
        self.selected_idx = torch.as_tensor(
            self.selected_joints,
            dtype=torch.long,
            device=self.tensor_args.device
        )  # shape [J]

    def forward(self, joint_state: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Compute joint bending cost.

        Args:
            joint_state: Tensor of shape [B, S, DOF].
            debug: If True, prints debug info.

        Returns:
            final_cost: Tensor of shape [B, S], bending cost per seed.
        """
        # Select the angles of interest: [B, S, J]
        group = joint_state.index_select(dim=-1, index=self.selected_idx)

        # Compute squared difference: (theta - target)**2
        # self.target_angles is [1,1,J], broadcast to [B,S,J]
        diff = group - self.target_angles
        sq   = diff.pow(2)

        # Apply per-joint weights if provided
        if self.joint_weights is not None:
            sq = sq * self.joint_weights  # broadcast

        # Sum over joints and apply global weight: [B, S]
        cost = sq.sum(dim=-1)
        final_cost = cost * self.weight

        # Debug output: show first few seeds
        if debug:
            sample = final_cost[0, :min(final_cost.size(1), 5)].tolist()
            print("=== [JointBending Debug] ===")
            # print selected joint indices
            print("Selected joint indices:", self.selected_idx.tolist())
            # print selected joint values for first batch, first seed
            print("Selected joint values (batch0, seed0):", group[0, 0, :].tolist())
            if self.joint_index_to_name:
                names = [self.joint_index_to_name[i] for i in self.selected_idx.tolist()]
                print("Joint names:", names)
            print("Target angles (rad):", self.target_angles.flatten().tolist())
            print("Sample cost per seed:", sample)

        return final_cost