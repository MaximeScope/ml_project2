import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from src import data_loader, model_cls, train, test, plotting, submissions

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ===== Torch config =====
    device = cfg.device
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.tensor_dtype))

    # ===== Data Loading =====
    train_loader, test_loader = data_loader.get_loader(cfg)

    # ===== Model, Optimizer and Loss function =====
    model = model_cls.Model()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.functional.cross_entropy

    # ===== Train Model =====
    train_losses, train_accs = train.train_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        cfg,
    )

    # ===== Test Model =====
    test_losses, test_accuracies = test.test_model(
        model,
        device,
        test_loader,
        loss_fn=partial(loss_fn, reduction="none"),
    )

    # ===== Preditions =====
    predictions = submissions.get_predictions(model, test_loader, cfg)

    # ==== Make Submission =====
    submissions.make_submission(predictions)

    # ===== Plotting =====
    plotting.plot_pred_on(test_loader, predictions, 1)

if __name__ == "__main__":
    run()