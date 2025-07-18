rm -r output/debug_learn_shadow
python src/main.py hand=shadow exp_name=debug_learn task=format task.max_num=-1 task.data_name=batched task.data_path=../LearnDexGrasp/output/experiment/shadow_floating_flow/tests/step_050000
python src/main.py hand=shadow task=eval exp_name=debug_learn task.max_num=-1