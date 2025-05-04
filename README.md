# Human-Like Grasp Synthesis (based on BODex)

This repository is a semester project based on [BODex](https://github.com/JYChen18/BODex), developed at TUM for exploring human-inspired dexterous grasp synthesis.  
We build upon the efficient GPU-based grasping pipeline from BODex, and **extend it to synthesize human-like grasp poses**, mimicking common grasp types such as two-finger pinch, three-finger tripod, and five-finger hook grasps.

> 🔧 This project is under active development and mainly targets learning, prototyping, and validation of task-oriented grasp synthesis resembling human strategies.

Original BODex: [Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490) ｜ [Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [Benchmark code](https://github.com/JYChen18/DexGraspBench)

---

## Introduction
### What’s New?
- **Human-like grasp types**: Focus on modeling and generating grasps inspired by common human strategies.
- **Custom constraints**: Implement constraints (e.g. finger coordination) to achieve grasp symmetry or shape consistency.
- **Hook grasp support**: New constraints are added to facilitate grasps suitable for handles and elongated objects.

---

*The rest of the original BODex README follows here...*

```
