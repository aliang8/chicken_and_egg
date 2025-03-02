import hydra
from omegaconf import DictConfig

from chicken_and_egg.trainers.fete_trainer import FETETrainer


@hydra.main(version_base=None, config_path="cfg", config_name="base")
def main(cfg: DictConfig):
    if cfg.name == "fete":
        trainer = FETETrainer(cfg)
    else:
        raise ValueError(f"Trainer {cfg.name} not found")

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
