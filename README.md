**Human-Like Grasp Synthesis (Ongoing | 2025.03 – Present)**

This repository contains my semester project at the Technical University of Munich (TUM), built on top of BODex. The goal of this project is to explore Human-like Robotic Dexterous Grasp Synthesis using BODex’s efficient GPU-based pipeline. Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, such as two-finger pinch, three-finger tripod, and five-finger hook, lumbrical, and spherical grasps.

 **Project Focus**
This project is under active development, primarily emphasizing the learning, prototyping, and validation of task-oriented grasp synthesis methods inspired by human grasping behaviors.

 **Energy Extensions**
To better replicate human grasp characteristics, I introduce joint-level constraints and energy terms:

* **Joint Consistency Energy:** Ensures coordinated finger joint movements within a group, facilitating realistic poses like hook and lumbrical grasps.
* **Joint Bending Energy:** Promotes natural finger flexion patterns aligned with specific human grasp types.

These customized energy components guide optimization toward structured, stable, and human-like grasp poses that maintain force closure.

**Final Goal**
Ultimately, the aim is to generate a large-scale, high-quality grasp dataset based on different human-like grasp types to support future research on data-driven robotic grasping methods.

Original BODex Resources:  
[📄 Project page](https://pku-epic.github.io/BODex) ｜ [📑 Paper](https://arxiv.org/abs/2412.16490) ｜ [🗃️ Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [💻 Benchmark code](https://github.com/JYChen18/DexGraspBench)
### 📸 Grasp Examples

- **Lumbrical Grasp Examples**  
  <br>
  <img src="https://github.com/user-attachments/assets/051551ca-5cf1-427d-9445-fe148e50008b" width="400"/>

- **Power Grasp Examples**  
  <br>
  <img src="https://github.com/user-attachments/assets/8cd0dfc2-358a-4caf-96bc-5342d1da5bdb" width="400"/>

- **Two-Finger Grasp Examples**  
  <br>
  <img src="https://github.com/user-attachments/assets/0757d264-2901-46ac-911b-318110bdf8c4" width="400"/>
- **spherical Grasp Examples**  
  <br>
  <img src="https://github.com/user-attachments/assets/8c947e39-9d69-48dc-877b-d0ecf7833c78" width="400"/>

- **cylindrical Grasp Examples**  
  <br>
  <img src="https://github.com/user-attachments/assets/749aa4ef-7471-4796-94f3-23beea50f81c" width="400"/>
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
## 5. Cylindrical Grasp

CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_hook.yml -w 20


