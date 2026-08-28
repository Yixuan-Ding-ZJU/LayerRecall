# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import (
    DiffusionTrainer,
    ScoreDistillationTrainer,
    CHPMTrainer,
)
from utils.config import normalize_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--resume-mode",
        choices=("none", "auto", "explicit"),
        default="auto",
    )
    parser.add_argument("--resume-checkpoint", type=str, default="")

    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    config.no_save = args.no_save

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = not args.enable_wandb
    config.auto_resume = args.resume_mode == "auto"
    config.resume_checkpoint = (
        args.resume_checkpoint if args.resume_mode == "explicit" else ""
    )
    if args.resume_mode == "explicit" and not args.resume_checkpoint:
        parser.error("--resume-mode explicit requires --resume-checkpoint")

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "chpm":
        trainer = CHPMTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
