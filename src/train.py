import torch.nn as nn

from src import utils
from tqdm import tqdm


def train_model(model, optimizer, loss_fn, train_loader, epochs, cfg):
    # Train the model over the specified number of epochs
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
    up = nn.Upsample(size=(608, 608), mode='bilinear')
    # Train every sample in the dataset
    for batch_idx, (img_batch, gt_batch) in enumerate(tqdm(train_loader), start=1):
        img_batch = img_batch.to(cfg.device)
        gt_batch = gt_batch.to(cfg.device)
        # Up-sample if the flag is set
        if cfg.training.upsample_to_test_size:
            img_batch = up(img_batch) 
            gt_batch = up(gt_batch.unsqueeze(1)).squeeze(1)

        output = model(img_batch)

        # Compute the loss and F1 over the training data
        loss = loss_fn(output, gt_batch)
        f1 = utils.get_f1(output, gt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append losses and F1 score to the list
        loss_float = loss.item()
        batch_losses.append(loss_float)
        batch_f1s.append(f1)
    avg_loss = sum(batch_losses) / len(batch_losses)
    avg_f1 = sum(batch_f1s) / len(batch_f1s)
    # Return average loss and F1 score
    return avg_loss, avg_f1
