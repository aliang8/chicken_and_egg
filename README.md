# Chicken and Egg

A reinforcement learning package for exploring exploration strategies in multi-armed bandit environments.

## Installation

You can install the package directly from the repository:

```bash
conda create -n chicken_and_egg python=3.10
conda activate chicken_and_egg
pip install -e .
```

## Local Config

Make a file called in the config directory named cfg/local/default.yaml with the following content.

```
# @package _global_

paths:
  root_dir: `path_to_your_repo/chicken_and_egg`

wandb:
  entity: `your_wandb_entity`
  project: `your_wandb_project`
```

## Running FETE training

To run the training script, use the following command:

```bash
python main.py --config-name=config
```

