from copy import deepcopy
import numpy as np
import random
import transforms3d

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from curobo.util_file import (
    load_yaml,
    load_json,
    get_world_configs_path,
    get_assets_path,
    get_output_path,
    join_path,
    get_pathlist_from_template,
)


class WorldConfigDataset(Dataset):

    def __init__(self, template_path, indicator, selection={}, additional_cfg_path=None):
        self.indicator = indicator

        template_path = join_path(get_assets_path(), template_path)
        self.full_path_list, self.id_list = get_pathlist_from_template(
            template_path,
            indicator=self.indicator,
            **selection,
        )

        if len(self.full_path_list) == 0:
            raise NotImplementedError(f"Cannot find any file with {template_path}")

        if additional_cfg_path is not None:
            example_path = join_path(get_world_configs_path(), additional_cfg_path)
            self.world_cfg_example = load_yaml(example_path)
            if "mesh" not in self.world_cfg_example:
                self.world_cfg_example["mesh"] = {}
        else:
            self.world_cfg_example = {"mesh": {}}

        return

    def __len__(self):
        return len(self.full_path_list)

    def _replace_indicator(self, template: str, id: str):
        return template.replace(self.indicator, id)

    def __getitem__(self, index):
        full_path = self.full_path_list[index]
        id = self.id_list[index]
        return full_path, id


class NPYBasedConfigDataset(WorldConfigDataset):
    def __init__(self, template_path, indicator, selection={}, additional_cfg_path=None, **kargs):
        selection = {}
        super().__init__(template_path, indicator, selection, additional_cfg_path)

    def __getitem__(self, index):
        full_path, id = super().__getitem__(index)
        cfg = np.load(full_path, allow_pickle=True).item()
        for k, v in cfg.items():
            cfg[k] = v[0]
        cfg["save_prefix"] = id
        return cfg


class OBJBasedConfigDataset(WorldConfigDataset):
    def __init__(
        self,
        template_path,
        indicator,
        selection={},
        additional_cfg_path=None,
        base_info_path=None,
        urdf_path=None,
        pose_path=None,
        fixed_pose_lst=None,
        fixed_scale_lst=None,
    ):
        super().__init__(template_path, indicator, selection, additional_cfg_path)
        self.repeat_num = selection['repeat']
        self.base_info_path = base_info_path
        self.urdf_path = urdf_path
        self.pose_path = pose_path
        self.fixed_pose_lst = fixed_pose_lst
        self.fixed_scale_lst = fixed_scale_lst
        if self.fixed_scale_lst is None:
            raise NotImplementedError("Only support fixed scale lst")
        return

    def __getitem__(self, index):
        full_path, id = super().__getitem__(index)
        repeat_id = index % self.repeat_num 
        each_scale_num = self.repeat_num // len(self.fixed_scale_lst) + 1 - (self.repeat_num % len(self.fixed_scale_lst)==0)
        scale_id = repeat_id // each_scale_num
        pose_id = repeat_id % each_scale_num 
        obj_scale = self.fixed_scale_lst[scale_id]
        
        if self.fixed_pose_lst is not None:
            pose_num = pose_id % len(self.fixed_pose_lst)
            obj_pose = self.fixed_pose_lst[pose_num]
        else:
            pose_path = join_path(
                get_assets_path(), self._replace_indicator(self.pose_path, id)
            )
            obj_pose_lst = load_json(pose_path)
            pose_num = pose_id % len(obj_pose_lst)
            obj_pose = obj_pose_lst[pose_num]

        if self.fixed_pose_lst is None:
            obj_pose[2] *= obj_scale
            obj_pose[0] = 0.1 * random.random() + 0.7
            obj_pose[1] = 0.

        obj_scale = np.around(obj_scale, 2)
        pose_name = str(pose_id).zfill(3)
        scale_name = str(int(obj_scale * 100)).zfill(3)
        manip_name = f"{id}_scale{scale_name}"  # This is for warp

        world_cfg = deepcopy(self.world_cfg_example)
        world_cfg["mesh"][manip_name] = {
            "scale": obj_scale,
            "pose": obj_pose,
            "file_path": full_path,
            "urdf_path": self._replace_indicator(self.urdf_path, id),
        }

        base_info_path = join_path(
            get_assets_path(), self._replace_indicator(self.base_info_path, id)
        )
        json_data = load_json(base_info_path)
        obj_rot = transforms3d.quaternions.quat2mat(obj_pose[3:])
        gravity_center = obj_pose[:3] + obj_rot @ json_data["gravity_center"] * obj_scale
        obb_length = obj_scale * np.linalg.norm(json_data["obb"]) / 2

        return {
            "world_cfg": world_cfg,
            "obj_code": id,
            "manip_name": id + f"_scale{scale_name}",
            "obj_path": full_path,
            "obj_scale": obj_scale,
            "obj_pose": np.array(obj_pose),
            "obj_gravity_center": gravity_center,
            "obj_obb_length": obb_length,
            "save_prefix": f"{id}/scale{scale_name}_pose{pose_name}_",
        }


def _world_config_collate_fn(list_data):
    if "world_cfg" in list_data[0]:
        world_cfg_lst = [i.pop("world_cfg") for i in list_data]
    else:
        world_cfg_lst = None
    ret_data = default_collate(list_data)
    if world_cfg_lst is not None:
        ret_data["world_cfg"] = world_cfg_lst
    return ret_data


def get_world_config_dataloader(configs, batch_size):
    if configs["template_path"].endswith(".obj"):
        dataset = OBJBasedConfigDataset(**configs)
    elif configs["template_path"].endswith(".npy"):
        dataset = NPYBasedConfigDataset(**configs)
    else:
        raise NotImplementedError("Unsupported file type.")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_world_config_collate_fn
    )
    return dataloader
