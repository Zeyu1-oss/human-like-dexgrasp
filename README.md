# Human-Like Grasp Synthesis (Ongoing | 2025.03 – Present)

This repository contains my semester project at the Technical University of Munich (TUM), built on top of [BODex](https://pku-epic.github.io/BODex). The goal of this project is to explore **human-like robotic dexterous grasp synthesis** using BODex’s efficient GPU-based pipeline. Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, including:

- Two-finger pinch  
- Three-finger tripod  
- Five-finger hook  
- Lumbrical grasp  
- Cylindrical grasp  

## 🔬 Project Focus

This project is under active development, with an emphasis on learning, prototyping, and validating **task-oriented grasp synthesis** methods inspired by human hand behavior.

## ⚙️ Energy Extensions

To better replicate human grasp characteristics, I introduced two customized energy components into the optimization process:

- **Joint Consistency Energy**  
  Ensures coordinated movement within finger joint groups, enabling realistic poses like *hook* and *lumbrical* grasps.

- **Joint Bending Energy**  
  Promotes natural finger flexion patterns aligned with specific human grasp styles.

These components guide optimization toward structured, stable, and human-like grasp poses that maintain **force closure**.

## 🎯 Final Goal

The final goal is to generate a large-scale, high-quality grasp dataset based on various human-like grasp types to support future research in **data-driven robotic grasping**.

---

## 📊 Grasp Dataset Summary

For each grasp type, I generated **200 grasp samples**, visualized and stored using `plan_batch_env.py`. The grasp outcomes are organized and accessible via the following link:

🔗 **[Grasp Dataset Visualization (Google Drive)](https://drive.google.com/drive/folders/1NrTXjJ25SCxDgjDlmIk2513UGFA6zsBh?usp=drive_link)**

| Grasp Type        | # Samples | Verified Success Rate       | Evaluation Method             |
|-------------------|-----------|------------------------------|--------------------------------|
| Power Grasp       | 200       | ✅ **100%**                   | Manual inspection + simulation |
| Pinch Grasp       | 200       | ✅ **100%**                   | Manual inspection + simulation |
| Tripod Grasp      | 200       | ✅ **100%**                   | Manual inspection + simulation |
| Lumbrical Grasp   | 200       | ✅ **≥ 70%**                  | Visual inspection + simulation |
| Cylindrical Grasp | 200       | ✅ **≥ 70%**                  | Visual inspection + simulation |

> ✅ *Power, Pinch, and Tripod* grasps were manually verified to achieve near-perfect stability and force closure.  
> ⚠️ *Lumbrical and Cylindrical* types are inherently more challenging but show stable performance above 70%.

---

## 🖼️ Grasp Examples

- **Lumbrical Grasp**  
  <img src="https://github.com/user-attachments/assets/051551ca-5cf1-427d-9445-fe148e50008b" width="400"/>

- **Power Grasp**  
  <img src="https://github.com/user-attachments/assets/8cd0dfc2-358a-4caf-96bc-5342d1da5bdb" width="400"/>

- **Two-Finger Grasp**  
  <img src="https://github.com/user-attachments/assets/0757d264-2901-46ac-911b-318110bdf8c4" width="400"/>

- **Cylindrical Grasp**  
  <img src="https://github.com/user-attachments/assets/e9580c92-64c8-4c1c-b2c1-17934c6ac4a6" width="400"/>

---

## 🛠️ Run Grasp Generation

```bash
# 1. Generate Lumbrical Grasp
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_lumbrical.yml -w 20

# 2. Generate Power Grasp
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_power.yml -w 20

# 3. Generate Two-Finger Grasp
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_2finger.yml -w 20

# 4. Generate Tripod Grasp
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_3finger.yml -w 20

# 5. Generate Cylindrical Grasp
CUDA_VISIBLE_DEVICES=0 python example_grasp/plan_batch_env.py -c sim_shadow/fc_hook.yml -w 20

