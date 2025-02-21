import hydra
from omegaconf import DictConfig

from chicken_and_egg.trainers.fete_trainer import FETETrainer


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    # Create trainer
    trainer = FETETrainer(cfg)

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()
