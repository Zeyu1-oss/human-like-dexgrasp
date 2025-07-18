# task_mergeobj.py

import os
import numpy as np
from glob import glob
import trimesh
from util.hand_util import RobotKinematics
import logging


def task_mergeobj(configs):
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_cpoint, *list(configs.data_struct)))

    if configs.task.vis_type == "succ":
        data_folder = configs.succ_dir
        input_file_lst = succ_lst
    elif configs.task.vis_type == "fail":
        data_folder = configs.eval_dir
        input_file_lst = list(
            set(eval_lst).difference(
                set([p.replace(configs.succ_dir, configs.eval_dir) for p in succ_lst])
            )
        )
    elif configs.task.vis_type == "raw":
        data_folder = configs.grasp_dir
        input_file_lst = grasp_lst
    else:
        raise NotImplementedError

    input_file_lst = sorted(input_file_lst)
    if configs.task.max_num > 0:
    # k 不超过现有文件数
        k = min(configs.task.max_num, len(input_file_lst))
        input_file_lst = np.random.choice(
             input_file_lst,
             size=k,
            replace=False)



    print(f"[INFO] Merging {len(input_file_lst)} grasp-object pairs into a grid")

    merged_scene = []
    x_offset = 0.0
    z_offset = 0.0
    base_spacing = 0.04 * 3  # 每组间水平/垂直间隔放大为 12cm
    row_max = 10         # 每行10组

    for i, path in enumerate(input_file_lst):
        grasp_data = np.load(path, allow_pickle=True).item()
        hand_fk = RobotKinematics(configs.hand.xml_path)

        all_qpos = grasp_data["grasp_qpos"]
        if len(all_qpos.shape) == 1:
            all_qpos = all_qpos[None]
        if configs.hand.mocap:
            hand_pose = all_qpos[0, :7]
            hand_qpos = all_qpos[0, 7:]
        else:
            hand_pose = np.array([0.0, 0, 0, 1, 0, 0, 0])
            hand_qpos = all_qpos[0]

        hand_fk.forward_kinematics(hand_qpos)
        hand_tm = hand_fk.get_posed_meshes(hand_pose)

        obj_path = os.path.join(grasp_data["obj_path"], "mesh/coacd.obj")
        obj_tm = trimesh.load(obj_path, force="mesh")
        obj_tm.vertices *= grasp_data["obj_scale"]
        T = trimesh.transformations.quaternion_matrix(grasp_data["obj_pose"][3:])
        T[:3, 3] = grasp_data["obj_pose"][:3]
        obj_tm.apply_transform(T)

        # 合并组合边界计算 spacing
        pair_bbox = np.vstack([hand_tm.bounds, obj_tm.bounds])
        pair_width = pair_bbox[:, 0].max() - pair_bbox[:, 0].min()
        spacing = pair_width + base_spacing

        # 应用平移 (在 x-z 平面上排布)
        translation = [x_offset, 0, z_offset]
        hand_tm.apply_translation(translation)
        obj_tm.apply_translation(translation)

        merged_scene.append(obj_tm)
        merged_scene.append(hand_tm)

        x_offset += spacing
        if (i + 1) % row_max == 0:
            x_offset = 0.0
            z_offset -= spacing

    merged_mesh = trimesh.util.concatenate(merged_scene)
    os.makedirs(configs.vobj_dir, exist_ok=True)
    output_path = os.path.join(configs.vobj_dir, "merged_grasps_grid.obj")
    merged_mesh.export(output_path)

    print(f"[✔️] Merged grasp-object grid saved at: {output_path}")
    logging.info(f"[✔️] Merged grasp-object grid saved at: {output_path}")
