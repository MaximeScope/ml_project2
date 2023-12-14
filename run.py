import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from src import data_loader, train, test, plotting, submissions, unet, utils
import mask_to_submission


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ===== Torch config =====
    device = cfg.device
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.tensor_dtype))

    # ===== Data Loading =====
    train_loader = data_loader.get_loader(cfg)

    plotting.plot_random_sample(train_loader, indices=[109, 293])

    # ===== Model, Optimizer and Loss function =====
    model = unet.UNet(cfg)
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loss_fn = torch.nn.functional.binary_cross_entropy

    #train.optimize_param(loss_fn, 5, "lr", np.logspace(-3.6, -2.6, 5), 1e-4, cfg)
    # ===== Train Model =====
    train_losses, train_f1s = train.train_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        cfg.training.epochs,
        cfg,
    )

    # ===== Test Model =====
    # test_loss, test_f1 = test.test_model(
    #     model,
    #     device,
    #     test_loader,
    #     loss_fn=partial(loss_fn, reduction="none"),
    # )

    # ==== Make Submission =====
    test_loader = data_loader.get_test_loader(cfg)
    predictions = submissions.get_predictions(model, test_loader, cfg)
    img_filenames = submissions.save_prediction_masks(predictions, test_loader, "predictions")
    #patched_preds = submissions.make_submission(predictions, cfg)
    mask_to_submission.masks_to_submission("submission.csv", *img_filenames)

    # ===== Plotting =====
    plotting.plot_train(train_losses, train_f1s)
    plotting.plot_pred_on(test_loader, predictions, 1, cfg)
    #plotting.plot_pred_on(test_loader, patched_preds, 1, cfg)


if __name__ == "__main__":
    run()
