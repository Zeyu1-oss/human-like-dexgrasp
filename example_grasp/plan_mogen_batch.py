# Standard Library
import time
from typing import Dict, List, Union
import os 
import datetime 

# Third Party
import torch
import numpy as np
import argparse

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.save_helper import SaveHelper
from curobo.util.logger import setup_logger, log_warn
from curobo.util_file import (
    get_manip_configs_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import random 
seed = 123 
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def init_motion_gen(robot_cfg: Dict, 
                    world_model: Dict, 
                    manip_name_list: Union[str, List[str]],
                    collision_activation_distance: float,
                    batch: int,
                    ):
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_model,
        interpolation_steps=1000,
        contact_obj_names=manip_name_list,
        collision_activation_distance=collision_activation_distance,
        ik_opt_iters=200,
        grad_trajopt_iters=200,
    )
    motion_gen = MotionGen(motion_gen_cfg)
    # motion_gen.warmup(batch=batch, warmup_link_poses=True)
    return motion_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default='fc_leap.yml',
        help="config file path",
    )
    
    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder",
    )
    
    parser.add_argument(
        "-m",
        "--save_mode",
        choices=['usd', 'npy', 'usd+npy', 'none'],
        default='npy',
        help="Method to save results",
    )
    
    parser.add_argument(
        "-t",
        "--task",
        choices=['grasp', 'mogen', 'grasp_and_mogen'],
        default='grasp_and_mogen',
    )
    
    parser.add_argument(
        "-k",
        "--skip",
        action='store_false',
        help="If True, skip existing files. (default: True)",
    )
    
    args = parser.parse_args()

    setup_logger("warn")
    tensor_args = TensorDeviceType()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))
    
    if args.save_folder is not None:
        save_folder = os.path.join(args.save_folder, 'graspdata')
    elif manip_config_data['exp_name'] is not None:
        save_folder = os.path.join(args.manip_cfg_file[:-4], manip_config_data['exp_name'], 'graspdata')
    else:
        save_folder = os.path.join(args.manip_cfg_file[:-4], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), 'graspdata')
    
    if 'grasp' in args.task:
        log_warn('Generate grasp pose from scratch!')
    else:
        manip_config_data['world']['template_path'] = os.path.join(save_folder, manip_config_data['world']['indicator'] + 'grasp.npy')
        log_warn(f"Load grasp pose from {manip_config_data['world']['candidate']['template_npy_path']}!")
   
    world_generator = get_world_config_dataloader(manip_config_data['world'], 1)
    
    save_grasp = SaveHelper(
                    robot_file=manip_config_data['robot_file'],
                    save_folder=save_folder,
                    task_name='grasp',
                    mode=args.save_mode,
                )
    
    save_mogen = SaveHelper(
                    robot_file=manip_config_data['robot_file_with_arm'],
                    save_folder=save_folder,
                    task_name='mogen',
                    mode=args.save_mode,
                )
    tst = time.time()
    mg = None 
    grasp_solver = None 
    for world_info_dict in world_generator:
        sst = time.time()
        if args.skip and (('mogen' not in args.task and save_grasp.exist_piece(world_info_dict['save_prefix'])) or save_mogen.exist_piece(world_info_dict['save_prefix'])):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue
        
        world_model = [WorldConfig.from_dict(world_info_dict['world_cfg'][0])]

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                    world_model=world_model,
                    manip_name_list=world_info_dict['manip_name'],
                    manip_config_data=manip_config_data,
                    obj_gravity_center=world_info_dict['obj_gravity_center'],
                    obj_obb_length=world_info_dict['obj_obb_length'],
                    use_cuda_graph=False,
                )
            grasp_solver = GraspSolver(grasp_config)
        else:
            grasp_solver.update_world(world_model, world_info_dict['obj_gravity_center'], world_info_dict['obj_obb_length'], world_info_dict['manip_name'])
        
        # plan grasp
        if 'robot_pose' not in world_info_dict.keys():
            grasp_result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)
            world_info_dict['world_model'] = world_model
            squeeze_pose_qpos = torch.cat([grasp_result.solution[..., 1, :7], grasp_result.solution[..., 1, 7:] * 2 - grasp_result.solution[..., 0, 7:]], dim=-1)
            all_hand_pose_qpos = torch.cat([grasp_result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2)
            world_info_dict['robot_pose'] = all_hand_pose_qpos   # [b, n, q]
            world_info_dict['contact_point'] = grasp_result.contact_point   # [b, n, p, 3]
            world_info_dict['contact_frame'] = grasp_result.contact_frame    # [b, n, p, 3, 3]
            world_info_dict['contact_force'] = grasp_result.contact_force   # [b, n, m, p, 3]
            world_info_dict['grasp_error'] = grasp_result.grasp_error   # [b, n, m]
            world_info_dict['dist_error'] = grasp_result.dist_error
            save_grasp.save_piece(world_info_dict)
            all_hand_pose_qpos = all_hand_pose_qpos.reshape((-1,) + all_hand_pose_qpos.shape[2:])
            smt = time.time()
            log_warn(f'Sinlge Time (grasp): {smt-sst}')
        else:
            all_hand_pose_qpos = tensor_args.to_device(world_info_dict['robot_pose'][0])
            smt = time.time()
        
        if 'mogen' not in args.task:
            continue 
        
        if mg is None:
            mg = init_motion_gen(manip_config_data['robot_file_with_arm'], 
                                world_model, 
                                world_info_dict['manip_name'], 
                                manip_config_data['grasp_contact_strategy']['distance'][1], 
                                all_hand_pose_qpos.shape[0]
                            )
        else:
            mg.update_world(world_model[0], world_info_dict['manip_name'])
        
        # plan trajectory
        pregrasp_pose_qpos, grasp_pose_qpos = all_hand_pose_qpos[:, 0, :], all_hand_pose_qpos[:, 1, :]
        pregrasp_hand_qpos, grasp_hand_qpos, squeeze_hand_qpos = all_hand_pose_qpos[:, 0, 7:], all_hand_pose_qpos[:, 1, 7:], all_hand_pose_qpos[:, 2, 7:]
        kin_state = grasp_solver.fk(pregrasp_pose_qpos)
        init_state = JointState.from_position(tensor_args.to_device(manip_config_data['mogen_init']).view(1, -1).repeat(all_hand_pose_qpos.shape[0], 1))
        mogen_result = mg.plan_batch(init_state, kin_state.ee_pose, link_poses=kin_state.link_poses, plan_config=MotionGenPlanConfig(enable_finetune_trajopt=False, num_trajopt_seeds=4, max_attempts=1))
        log_warn('MoGen Success rate: %.2f' % (mogen_result.success.int().sum() / len(mogen_result.success)))
        log_warn(f'MoGen Failure: {torch.where(mogen_result.success == 0)[0]}')
        robot_pose_traj = [mogen_result.optimized_plan.position]
        
        # grasp qpos
        pregrasp_arm_qpos = mogen_result.optimized_plan.position[:, -1, :grasp_solver.ik_solver.kinematics.dof]
        kin_state = grasp_solver.fk(grasp_pose_qpos)
        ik_result = grasp_solver.ik_solver.solve_batch(kin_state.ee_pose, seed_config=pregrasp_arm_qpos.unsqueeze(1), retract_config=pregrasp_arm_qpos)
        log_warn('IK for Grasp Pose Success rate: %.2f' % (ik_result.success.int().sum() / len(ik_result.success)))
        log_warn(f'IK for Grasp Pose Failure: {torch.where(ik_result.success == 0)[0]}')
        robot_pose_traj.append(torch.cat([ik_result.solution, grasp_hand_qpos.unsqueeze(1)], dim=-1))

        # squeeze qpos
        robot_pose_traj.append(torch.cat([ik_result.solution, squeeze_hand_qpos.unsqueeze(1)], dim=-1))
        
        # lift pose
        kin_state = grasp_solver.fk(grasp_pose_qpos)
        kin_state.ee_pose.position[..., -1] += 0.1
        ik_result = grasp_solver.ik_solver.solve_batch(kin_state.ee_pose, seed_config=pregrasp_arm_qpos.unsqueeze(1), retract_config=pregrasp_arm_qpos)
        log_warn('Lift IK Success rate: %.2f' % (ik_result.success.int().sum() / len(ik_result.success)))
        robot_pose_traj.append(torch.cat([ik_result.solution, squeeze_hand_qpos.unsqueeze(1)], dim=-1))
        robot_pose_traj = torch.cat(robot_pose_traj, dim=1)
        
        # save results
        world_info_dict['world_model'] = world_model
        world_info_dict['robot_pose'] = robot_pose_traj.unsqueeze(0)
        save_mogen.save_piece(world_info_dict)
        log_warn(f'Sinlge Time (mogen): {time.time()-smt}')
    log_warn(f'Total Time: {time.time()-tst}')
            