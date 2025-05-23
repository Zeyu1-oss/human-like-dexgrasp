#  Human-Like Grasp Synthesis (Based on BODex)

This repository contains my **semester project** at the Technical University of Munich (TUM), built on top of [BODex](https://github.com/JYChen18/BODex).  
The goal of this project is to explore **human-inspired dexterous grasp synthesis** using BODex’s efficient GPU-based pipeline.  
Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, such as two-finger pinch, three-finger tripod, and five-finger hook grasps.

> 🔧 This project is under active development, and mainly focuses on learning, prototyping, and validating **task-oriented grasp synthesis** approaches inspired by the way humans grasp objects.

Original BODex Resources:  
[📄 Project page](https://pku-epic.github.io/BODex) ｜ [📑 Paper](https://arxiv.org/abs/2412.16490) ｜ [🗃️ Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [💻 Benchmark code](https://github.com/JYChen18/DexGraspBench)

---

## 1. Generate Lumbrical Grasp

```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_lumbrical.yml -w 20

```
---

## 2. Generate Spherical Grasp

```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_spherical.yml -w 20

```
## 3. Random power Grasp
```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_power.yml -w 20

```
## 4. 2fingers pinch Grasp
```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_2finger.yml -w 20

```
## 5. tripod Grasp
```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_3finger.yml -w 20

```


