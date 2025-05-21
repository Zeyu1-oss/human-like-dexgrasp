# 🤖 Human-Like Grasp Synthesis (Based on BODex)

This repository contains my **semester project** at the Technical University of Munich (TUM), built on top of [BODex](https://github.com/JYChen18/BODex).  
The goal of this project is to explore **human-inspired dexterous grasp synthesis** using BODex’s efficient GPU-based pipeline.  
Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, such as two-finger pinch, three-finger tripod, and five-finger hook grasps.

> 🔧 This project is under active development, and mainly focuses on learning, prototyping, and validating **task-oriented grasp synthesis** approaches inspired by the way humans grasp objects.

Original BODex Resources:  
[📄 Project page](https://pku-epic.github.io/BODex) ｜ [📑 Paper](https://arxiv.org/abs/2412.16490) ｜ [🗃️ Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [💻 Benchmark code](https://github.com/JYChen18/DexGraspBench)

---

## 🧪 Example: Generate Lumbrical Grasp

```bash
python src/main.py\task=format\task.data_path=/home/rose/BODex/src/curobo/content/assets/output/sim_shadow/fc_lumbrical/debug/graspdata

