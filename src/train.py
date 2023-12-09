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
    # train_loader_extended = []

    # for (img_batch, gt_batch) in train_loader:
    #     train_loader_extended.append((img_batch, gt_batch))
    #     for amount in [1, 2, 3]:
    #         img_rot = utils.rotate_batch(img_batch, amount, dims=(2, 3))
    #         gt_rot = utils.rotate_batch(gt_batch, amount, dims=(1, 2))
    #         train_loader_extended.append((img_rot, gt_rot))

    # indices=[1, 2, 3, 4, 5]
    # # Set the figure size based on the number of samples
    # _, axes = plt.subplots(len(indices), 2, figsize=(8, 4))

    # for i, idx in enumerate(indices):
    #     # Get the sample using the generated index
    #     image, groundtruth = train_loader_extended[idx]

    #     if len(indices) == 1:
    #         # Plot the original image
    #         axes[0].imshow(image[0].permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    #         axes[0].set_title("Original Image")
    #         axes[0].axis("off")
    #         # Plot the ground truth
    #         axes[1].imshow(groundtruth[0], cmap="gray")
    #         axes[1].set_title("Ground Truth")
    #         axes[1].axis("off")
    #     else:
    #         axes[i, 0].imshow(
    #             image[0].permute(1, 2, 0)
    #         )  # Permute to (H, W, C) for plotting
    #         axes[i, 0].set_title("Original Image")
    #         axes[i, 0].axis("off")
    #         # Plot the ground truth
    #         axes[i, 1].imshow(groundtruth[0], cmap="gray")
    #         axes[i, 1].set_title("Ground Truth")
    #         axes[i, 1].axis("off")

    # plt.tight_layout()
    # plt.show()
    
    # # shuffle train_loader_extended
    # random.shuffle(train_loader_extended)

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
