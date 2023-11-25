import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import data_loader
import model_cls
import train
import test

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = cfg.device

    # ===== Data Loading =====
    train_loader, test_loader = data_loader.get_loader()

    # ===== Model, Optimizer and Loss function =====
    model = model_cls.Model()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.functional.cross_entropy

    # ===== Train Model =====


    # ===== Test Model =====
    losses, accuracies = test.test_model(
        model,
        device,
        test_loader,
        loss_fn=partial(loss_fn, reduction="none"),
    )

if __name__ == "__main__":
    run()