import hydra
from omegaconf import DictConfig, OmegaConf

from chicken_and_egg.trainers.fete_trainer import FETETrainer

OmegaConf.register_new_resolver("eval", eval, replace=True)


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
