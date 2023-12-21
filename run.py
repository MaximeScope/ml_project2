import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src import data_loader, train, plotting, submissions, unet
import mask_to_submission


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print('-' * 80)

    # ===== Torch config =====
    device = cfg.device
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.tensor_dtype))

    # ===== Data Loading =====
    train_loader = data_loader.get_loader(cfg)

    # ===== Model, Optimizer and Loss function =====
    model = unet.UNet(cfg)
    print(f'U-Net parameters: {sum(p.numel() for p in model.parameters())}')
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loss_fn = torch.nn.functional.binary_cross_entropy

    # ===== Train Model =====
    train_losses, train_f1s = train.train_model(
        model,
        optimizer,
        loss_fn,
        train_loader,
        cfg.training.epochs,
        cfg,
    )

    # ==== Make Submission =====
    test_loader = data_loader.get_test_loader(cfg)
    predictions = submissions.get_predictions(model, test_loader, cfg)
    img_filenames = submissions.save_prediction_masks(predictions, test_loader, cfg)
    mask_to_submission.masks_to_submission("submission.csv", cfg, *img_filenames)

    # ===== Plotting =====
    plotting.plot_train(train_losses, train_f1s)
    plotting.plot_pred_on(test_loader, predictions, 1, cfg)


if __name__ == "__main__":
    run()
