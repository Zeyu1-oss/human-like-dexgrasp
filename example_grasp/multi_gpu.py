import os 
import numpy as np
from copy import deepcopy
import argparse
from curobo.util_file import (
    get_manip_configs_path,
    get_output_path,
    join_path,
    load_yaml,
    write_yaml,
)
import subprocess
import multiprocessing
import datetime

def worker(gpu_id, task, manip_path, save_folder, output_path, save_mode, parallel_world):
    with open(output_path, 'w') as output_file:
        if task == 'grasp':
            subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu_id} python example_grasp/plan_batch_env.py -c {manip_path} -f {save_folder} -m {save_mode} -w {parallel_world}", shell=True, stdout=output_file, stderr=output_file)
        else:
            subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu_id} python example_grasp/plan_mogen_batch.py -c {manip_path} -f {save_folder} -m {save_mode} -t {task}", shell=True, stdout=output_file, stderr=output_file)


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
        "-t",
        "--task",
        choices=['grasp', 'mogen', 'grasp_and_mogen'],
        default='grasp',
    )
    
    parser.add_argument(
        "-m",
        "--save_mode",
        choices=['usd', 'npy', 'usd+npy', 'none'],
        default='npy',
    )
    
    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="parallel world num (only used when task=grasp)",
    )
    
    parser.add_argument(
        "-g",
        "--gpu",
        nargs='+',
        required=True,
        help='gpu id list'
    )
    args = parser.parse_args()
    
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))
    
    if manip_config_data['world']['selection']['id'] is not None:
        id_lst = deepcopy(manip_config_data['world']['selection']['id'])
        original_start = 0
        original_end = len(manip_config_data['world']['selection']['id'])
    else:
        original_start = deepcopy(manip_config_data['world']['selection']['start'])
        original_end = deepcopy(manip_config_data['world']['selection']['end'])
    all_obj_num = original_end - original_start
    obj_num_lst = np.array([all_obj_num // len(args.gpu)] * len(args.gpu))
    obj_num_lst[:(all_obj_num % len(args.gpu))] += 1
    assert obj_num_lst.sum() == all_obj_num
    
    p_list = []
    if manip_config_data['exp_name'] is not None:
        save_folder = os.path.join(args.manip_cfg_file[:-4],manip_config_data['exp_name'])
    else:
        save_folder = os.path.join(args.manip_cfg_file[:-4], datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    runinfo_folder = os.path.join(get_output_path(), save_folder, 'runinfo')
    os.makedirs(runinfo_folder, exist_ok=True) 
       
    # create tmp manip cfg files
    for i, gpu_id in enumerate(args.gpu):
        new_manip_path = join_path(runinfo_folder, str(i) + '_config.yml')
        if manip_config_data['world']['selection']['id'] is not None:
            ss = original_start + (obj_num_lst[:i]).sum()
            ee = original_start + (obj_num_lst[:(i+1)]).sum()
            manip_config_data['world']['selection']['id'] = id_lst[ss:ee]
        else:
            manip_config_data['world']['selection']['start'] = int(original_start + (obj_num_lst[:i]).sum())
            manip_config_data['world']['selection']['end'] = int(original_start + (obj_num_lst[:(i+1)]).sum())
        write_yaml(manip_config_data, new_manip_path)

        output_path = join_path(runinfo_folder, str(i) + '_output.txt')

        p = multiprocessing.Process(target=worker, args=(gpu_id, args.task, new_manip_path, save_folder, output_path, args.save_mode, args.parallel_world))
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)
        
    for p in p_list:
        p.join()    