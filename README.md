# Benchmark: Human-Like Grasp Evaluation

This branch contains a modified version of [DexGraspBench](https://github.com/JYChen18/DexGraspBench). It is adapted to evaluate human-like robotic grasp synthesis results generated in the `main` branch of this repository.

The benchmark enables large-scale analysis and visualization of open-loop grasp trajectories using physical simulation and analytic metrics.

## Key Features

- Evaluate grasp success via MuJoCo simulation
- Analyze force closure, contact consistency, penetration depth, and grasp diversity
- Visualize grasp results using `task=mergeobj`

## Modifications from Original DexGraspBench

This branch builds upon the official DexGraspBench implementation:  
https://github.com/JYChen18/DexGraspBench

It introduces the following structural and functional changes:

- Integrated with synthesized human-like grasps from the `main` branch
- Added `task=mergeobj` configuration to visualize sampled grasp subsets by type
- Modified grasp success criteria to require a minimum number of hand links in contact with the object
- Removed unrelated baseline comparison modules to streamline usage

## Grasp Visualization Workflow

To merge and visualize selected grasp results:

```bash
python src/main.py task=mergeobj 
