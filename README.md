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

## Training

To run the training script, use the following command:

```bash
# DREAM
python main.py --config-name=train_dream \
  env=darkroom \
  run_id=0000

# FETE
python main.py --config-name=train_fete \
  env=darkroom \
  run_id=0000 \
  skip_first_eval=True 


# FETE with wandb (e.g.)

python main.py --config-name=train_fete   env=darkroom   run_id=0000   skip_first_eval=True  +wandb.name='cae_fete_test' +wandb.notes='N/A' +wandb.tags="" +wandb.group_name="cae" ++use_wandb=True
```



# log entropy of action logits 

