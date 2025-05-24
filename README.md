**Human-Like Grasp Synthesis (Ongoing | 2025.03 – Present)**  

This repository contains my semester project at the Technical University of Munich (TUM), built on top of BODex.
The goal of this project is to explore Human-like Robotic Dexterous Grasp Synthesis using BODex’s efficient GPU-based pipeline.
Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, such as two-finger pinch, three-finger tripod, and five-finger hook, lumbrical, and spherical grasps.

🔧 This project is under active development, and mainly focuses on learning, prototyping, and validating task-oriented grasp synthesis approaches inspired by the way humans grasp objects.

🔍 Method Extension
To better capture the characteristics of human grasping, I introduce joint-level constraints and energy terms based on:

Joint Consistency energy: to ensure that finger joints within a group move in a coordinated way, supporting poses like hook and lumbrical grasps.

Joint Bending Energy: to encourage natural finger flexion patterns aligned with specific human grasp types.

These customized energy components guide the optimization towards more structured and human-like grasp poses while maintaining stability and force closure.
Original BODex Resources:  
[📄 Project page](https://pku-epic.github.io/BODex) ｜ [📑 Paper](https://arxiv.org/abs/2412.16490) ｜ [🗃️ Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [💻 Benchmark code](https://github.com/JYChen18/DexGraspBench)
### 📸 Grasp Examples

- **Lumbrical Grasp Example**  
  <br>
  <img src="https://github.com/user-attachments/assets/051551ca-5cf1-427d-9445-fe148e50008b" width="400"/>

- **Power Grasp Example**  
  <br>
  <img src="https://github.com/user-attachments/assets/8cd0dfc2-358a-4caf-96bc-5342d1da5bdb" width="400"/>

- **Two-Finger Grasp Example**  
  <br>
  <img src="https://github.com/user-attachments/assets/0757d264-2901-46ac-911b-318110bdf8c4" width="400"/>

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
## 4. two-finger Grasp
```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_2finger.yml -w 20

```
## 5. tripod Grasp
```bash
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_3finger.yml -w 20

```


