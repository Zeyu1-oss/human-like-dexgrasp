import time
from typing import List
import datetime
import os
import torch
import numpy as np
import argparse

from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger, log_warn
from curobo.util.save_helper import SaveHelper
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import random
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def process_grasp_result(result, save_debug, save_data, save_id):
    traj = result.debug_info['solver']['steps'][0]
    all_traj = torch.cat(traj, dim=1)
    batch, horizon = all_traj.shape[:2]

    if save_data == 'all':
        select_horizon_lst = list(range(0, horizon))
    elif 'select_' in save_data:
        part_num = int(save_data.split('select_')[-1])
        select_horizon_lst = list(range(0, horizon, horizon // (part_num-1)))
        select_horizon_lst[-1] = horizon - 1
    elif save_data == 'init':
        select_horizon_lst = [0]
    elif save_data == 'final' or save_data == 'final_and_mid':
        select_horizon_lst = [-1]
    else:
        raise NotImplementedError

    if save_id is None:
        save_id_lst = list(range(0, batch))
    elif isinstance(save_id, List):
        save_id_lst = save_id
    else:
        raise NotImplementedError

    save_traj = all_traj[:, select_horizon_lst]
    save_traj = save_traj[save_id_lst, :]

    if save_debug:
        n_num = torch.stack(result.debug_info['solver']['hp'][0]).shape[-2]
        o_num = torch.stack(result.debug_info['solver']['op'][0]).shape[-2]
        hp_traj = torch.stack(result.debug_info['solver']['hp'][0], dim=1).view(-1, n_num, 3)
        grad_traj = torch.stack(result.debug_info['solver']['grad'][0], dim=1).view(-1, n_num, 3)
        op_traj = torch.stack(result.debug_info['solver']['op'][0], dim=1).view(-1, o_num, 3)
        posi_traj = torch.stack(result.debug_info['solver']['debug_posi'][0], dim=1).view(-1, o_num, 3)
        normal_traj = torch.stack(result.debug_info['solver']['debug_normal'][0], dim=1).view(-1, o_num, 3)

        debug_info = {
            'hp': hp_traj,
            'grad': grad_traj * 100,
            'op': op_traj,
            'debug_posi': posi_traj,
            'debug_normal': normal_traj,
        }

        for k, v in debug_info.items():
            debug_info[k] = v.view((all_traj.shape[0], -1) + v.shape[1:])[:, select_horizon_lst]
            debug_info[k] = debug_info[k][save_id_lst, :]
            debug_info[k] = debug_info[k].view((-1,) + v.shape[1:])
    else:
        debug_info = None
        if save_data == 'final_and_mid':
            mid_robot_pose = torch.cat(result.debug_info['solver']['mid_result'][0], dim=1)
            mid_robot_pose = mid_robot_pose[save_id_lst, :]
            save_traj = torch.cat([mid_robot_pose, save_traj], dim=-2)

    return save_traj, debug_info

def normalize_quaternion(q):
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / (norm + 1e-8)

def run_grasp_pipeline(manip_cfg_file, save_folder, save_mode, save_data, save_id, save_debug, skip, parallel_world):
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), manip_cfg_file))
    world_generator = get_world_config_dataloader(manip_config_data['world'], batch_size=parallel_world)
    
    #save path
    if save_folder is not None:
        save_path = os.path.join(save_folder, manip_cfg_file[:-4], 'graspdata')
    elif manip_config_data['exp_name'] is not None:
        save_path = os.path.join(manip_cfg_file[:-4], manip_config_data['exp_name'], 'graspdata')
    else:
        save_path = os.path.join(manip_cfg_file[:-4], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), 'graspdata')

    save_helper = SaveHelper(
        robot_file=manip_config_data['robot_file'],
        save_folder=save_path,
        task_name='grasp',
        mode=save_mode,
    )

    grasp_solver = None
    tst = time.time()
    for world_info_dict in world_generator:
        sst = time.time()
        if skip and save_helper.exist_piece(world_info_dict['save_prefix']):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict['world_cfg'],
                manip_name_list=world_info_dict['manip_name'],
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict['obj_gravity_center'],
                obj_obb_length=world_info_dict['obj_obb_length'],
                use_cuda_graph=False,
                store_debug=save_debug,
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict['world_model'] = grasp_solver.world_coll_checker.world_model
        else:
            world_model = [WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict['world_cfg']]
            grasp_solver.update_world(world_model, world_info_dict['obj_gravity_center'], world_info_dict['obj_obb_length'], world_info_dict['manip_name'])
            world_info_dict['world_model'] = world_model

        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        if save_debug:
            robot_pose, debug_info = process_grasp_result(result, save_debug, save_data, save_id)
            world_info_dict['debug_info'] = debug_info
            world_info_dict['robot_pose'] = robot_pose.reshape((len(world_info_dict['world_model']), -1) + robot_pose.shape[1:])
        else:
            squeeze_pose_qpos = torch.cat([
                result.solution[..., 1, :7],
                result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:]
            ], dim=-1)

            squeeze_pose_qpos[..., 3:7] = normalize_quaternion(squeeze_pose_qpos[..., 3:7])
            all_hand_pose_qpos = torch.cat([result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2)
            world_info_dict['robot_pose'] = all_hand_pose_qpos
            world_info_dict['contact_point'] = result.contact_point
            world_info_dict['contact_frame'] = result.contact_frame
            world_info_dict['contact_force'] = result.contact_force
            world_info_dict['grasp_error'] = result.grasp_error
            world_info_dict['dist_error'] = result.dist_error

        log_warn(f'Single Time: {time.time()-sst:.2f}s')
        save_helper.save_piece(world_info_dict)

    log_warn(f'Total Time: {time.time()-tst:.2f}s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list", nargs='+', required=True, help="List of yml config files")
    parser.add_argument("--save_folder", type=str, default=None, help="Base folder to save all results")
    parser.add_argument("--save_mode", choices=['usd', 'npy', 'usd+npy', 'none'], default='npy', help="Save mode")
    parser.add_argument("--save_data", default='final_and_mid', help="Which data to save")
    parser.add_argument("--save_id", type=int, nargs='+', default=None, help="Which results to save")
    parser.add_argument("--save_debug", action='store_true', help="Whether to save debug info")
    parser.add_argument("--parallel_world", type=int, default=20, help="Number of parallel environments")
    parser.add_argument("--skip", action='store_false', help="Skip if already exists")

    args = parser.parse_args()
    setup_logger("warn")

    for cfg in args.config_list:
        log_warn(f"[START] Running: {cfg}")
        run_grasp_pipeline(cfg, args.save_folder, args.save_mode, args.save_data, args.save_id, args.save_debug, args.skip, args.parallel_world)
        log_warn(f"[DONE]  Completed: {cfg}\n")
