import hydra
from omegaconf import DictConfig, OmegaConf

from chicken_and_egg.trainers import TRAINER_TO_CLS

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: DictConfig):
    if cfg.name not in TRAINER_TO_CLS:
        raise ValueError(f"Trainer {cfg.name} not found")

    trainer = TRAINER_TO_CLS[cfg.name](cfg)

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
