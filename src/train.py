import torch.optim

from src import utils, data_loader, test, unet
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def optimize_param(loss_fn, iterations, param_to_optimize, param_vals, other_param, cfg):
    if param_to_optimize == "lr":
        weight_decay = other_param
    else:
        lr = other_param

    best_param_val = 0
    best_f1 = 0
    for param_val in param_vals:
        if param_to_optimize == "lr":
            lr = param_val
        else:
            weight_decay = param_val

        model = unet.UNet(cfg)
        model = model.to(device=cfg.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        f1_sum = 0
        for _ in range(iterations):
            train_loader, test_loader = data_loader.get_loader(cfg)
            train_model(model, optimizer, loss_fn, train_loader, 1, cfg)
            f1_score = test.test_model(model, cfg.device, test_loader)
            f1_sum += f1_score

        f1_avg = f1_sum / iterations
        print("for " + param_to_optimize + "=" + str(param_val) + " f1_score=" + str(f1_avg))
        if f1_avg > best_f1:
            best_f1 = f1_avg
            best_param_val = param_val

    print("best " + param_to_optimize + "=" + str(best_param_val))
    return best_param_val


def train_model(model, optimizer, loss_fn, train_loader, epochs, cfg):
    train_losses = []
    train_f1s = []
    for epoch in range(1, epochs + 1):
        avg_loss, avg_acc = train_epoch(model, optimizer, loss_fn, train_loader, cfg)
        train_losses.append(avg_loss)
        train_f1s.append(avg_acc)

        print(f"Train Epoch: {epoch}: " f"loss={avg_loss:0.2e}, " f"f1={avg_acc:0.3f}")
    return train_losses, train_f1s


def train_epoch(model, optimizer, loss_fn, train_loader, cfg):
    model.train()
    batch_losses = []
    batch_f1s = []

    for batch_idx, (img_batch, gt_batch) in enumerate(tqdm(train_loader), start=1):
        img_batch = img_batch.to(cfg.device)
        gt_batch = gt_batch.to(cfg.device)
        output = model(img_batch)
        loss = loss_fn(output, gt_batch)
        f1 = utils.get_f1(output, gt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_float = loss.item()
        batch_losses.append(loss_float)
        batch_f1s.append(f1)
    avg_loss = sum(batch_losses) / len(batch_losses)
    avg_f1 = sum(batch_f1s) / len(batch_f1s)
    return avg_loss, avg_f1
