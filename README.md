# Human-Like Grasp Synthesis (Ongoing | 2025.03 – Present)

This repository contains my semester project at the Technical University of Munich (TUM), built on top of [BODex](https://pku-epic.github.io/BODex).Here is my supervisor's email: qian.feng@tum.de.
The goal of this project is to explore **human-like robotic dexterous grasp synthesis** using BODex’s efficient GPU-based pipeline. Specifically, I extend the original framework to generate grasp poses that mimic common human strategies, including:

- Two-finger pinch  
- Three-finger tripod  
- random power 
- Lumbrical grasp  
- Cylindrical grasp  

across **three different robotic hand designs**.

##  Project Focus

This project is under active development, with an emphasis on learning, prototyping, and validating **task-oriented grasp synthesis** methods inspired by human grasping behaviors.

## ⚙ Energy Extensions

To better replicate human grasp characteristics, I introduced two customized energy components into the optimization process:

- **Joint Consistency Energy**  
  Ensures coordinated movement within finger joint groups, enabling realistic poses like *hook* and *lumbrical* grasps.

- **Joint Bending Energy**  
  Promotes natural finger flexion patterns aligned with specific human grasp styles.

These components guide optimization toward structured, stable, and human-like grasp poses that maintain **force closure**.

---

##  Large-Scale Grasp Dataset

Using the extended pipeline, I generated a total of **1.5 million grasps**:

> 🔹 3 robotic hands × 5 grasp types × ~100,000 grasps per configuration  
> = **1,500,000 grasp samples**

All data was generated using GPU-accelerated optimization with joint-level energy terms and constraint configurations tailored to each grasp strategy and hand structure.

---

##  Visual Grasp Evaluation

To verify the validity of the synthesized grasp styles, I randomly sampled **200 grasps per strategy** (from the full dataset) for visual inspection. Each sample was manually assessed to determine whether it matched the **expected human grasp pattern**.

🔗 **[Grasp dataset with Visualization Samples (Google Drive)](https://drive.google.com/drive/folders/1NrTXjJ25SCxDgjDlmIk2513UGFA6zsBh?usp=drive_link)**

| Grasp Type        | # Visualized Samples | Match to Expected Strategy          |
|-------------------|----------------------|-------------------------------------|
| Power Grasp       | 200                  |  **~100%** (fully match)           |
| Pinch Grasp       | 200                  |  **~100%** (fully match)           |
| Tripod Grasp      | 200                  |  **~100%** (fully match)           |
| Lumbrical Grasp   | 200                  |  **≥ 70%** (partially match)       |
| Cylindrical Grasp | 200                  |  **≥ 70%** (partially match)       |

>  *Power, Pinch, and Tripod* examples consistently matched their intended grasp type.  
> ⚠ *Lumbrical and Cylindrical* examples showed more variability but still reflected expected patterns in the majority of cases.
It should be noted that lumbrical and cylindrical grasp synthesis was performed across the entire object dataset, without pre-selecting objects based on geometric suitability. Although basic heuristics were applied to filter out clearly incompatible candidates, some ambiguous or borderline objects remained. For these cases, the optimizer may naturally converge toward spherical or power-like grasps, deviating from the intended strategy. This explains the moderate match rate (~70%) observed for these grasp types, even though most results still reflect key characteristics of human-like grasping.
In future work, selecting object subsets specifically aligned with each grasp type could improve optimization performance, allowing the synthesis process to better exploit structural assumptions and further enhance grasp quality and consistency.

##  Grasp Examples

- **Lumbrical Grasp**  
  <img src="https://github.com/user-attachments/assets/051551ca-5cf1-427d-9445-fe148e50008b" width="400"/>

- **Power Grasp**  
  <img src="https://github.com/user-attachments/assets/8cd0dfc2-358a-4caf-96bc-5342d1da5bdb" width="400"/>

- **Two-Finger Grasp**  
  <img src="https://github.com/user-attachments/assets/0757d264-2901-46ac-911b-318110bdf8c4" width="400"/>

- **Cylindrical Grasp**  
  <img src="https://github.com/user-attachments/assets/e9580c92-64c8-4c1c-b2c1-17934c6ac4a6" width="400"/>

---

##  Run Grasp Generation

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
