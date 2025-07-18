import os
import multiprocessing
import logging
from glob import glob
import traceback
import gc
import itertools

import numpy as np

from .eval_func import *

def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        if configs.hand.mocap:
            eval_func_name = f"{configs.setting}MocapEval"
        else:
            eval_func_name = f"{configs.setting}ArmEval"
        eval(eval_func_name)(input_npy_path, configs).run()
        return
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return




def task_eval(configs):
    assert (
        configs.task.simulation_metrics is not None
        or configs.task.analytic_fc_metrics is not None
        or configs.task.pene_contact_metrics is not None
    ), "You should at least evaluate one kind of metrics"

    # Gather input files
    input_path_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    init_num = len(input_path_lst)

    # Skip already evaluated
    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
        eval_path_lst = [p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst]
        input_path_lst = list(set(input_path_lst).difference(eval_path_lst))

    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)

    # Limit number of tasks if requested
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]

    logging.info(
        f"Find {init_num} grasp data in {configs.grasp_dir}, skip {skip_num}, and use {len(input_path_lst)}."
    )
    logging.info(f"[DEBUG] min_contact_num from config: {configs.task.min_contact_num}")

    if not input_path_lst:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    debug_mode = getattr(configs.task, "debug_mode", False)

    # Main evaluation
    results = []  # Initialize results list
    if debug_mode:
        print("[INFO] Debug mode is ON: Running in single-threaded mode for easier debugging.")
        for ip in iterable_params:
            results.append(safe_eval_one(ip))
            print()
    else:
        # Parallel execution with controlled memory usage
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(
                safe_eval_one,
                iterable_params,
                chunksize=1000
            )

            BATCH_SIZE = 2000
            while True:
                batch = list(itertools.islice(result_iter, BATCH_SIZE))
                if not batch:
                    break
                results.extend(batch)

    # After evaluation, update logs and lists
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    succ_cpoint_list = glob(os.path.join(configs.succ_cpoint, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))

    logging.info(f"Saved {len(succ_lst)} total successful grasps into {configs.succ_dir}")
    logging.info(
        f"Saved {len(succ_cpoint_list)} grasps with contact_num >= "
        f"{getattr(configs.task, 'min_contact_num', '?')} into {configs.succ_cpoint}"
    )
    logging.info(
        f"Get {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    logging.info("Finish evaluation")

    return results