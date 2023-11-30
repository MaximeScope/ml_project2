from src import utils
from tqdm import tqdm


def train_model(model, optimizer, loss_fn, train_loader, cfg):
    train_losses = []
    train_f1s = []
    for epoch in range(1, cfg.training.epochs + 1):
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
        img_batch.to(cfg.device)
        gt_batch.to(cfg.device)
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
