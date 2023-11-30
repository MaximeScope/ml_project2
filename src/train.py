from src import utils


def train_model(model, optimizer, loss_fn, train_loader, cfg):
    train_losses = []
    train_accs = []
    for epoch in range(1, cfg.training.epochs + 1):
        avg_loss, avg_acc = train_epoch(
            model, optimizer, loss_fn, train_loader, cfg
        )
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        print(
            f"Train Epoch: {epoch}"
            f"batch_loss={avg_loss:0.2e} "
            f"batch_acc={avg_acc:0.3f} "
        )
    return train_losses, train_accs


def train_epoch(model, optimizer, loss_fn, train_loader, cfg):
    model.train()
    batch_losses = []
    batch_accs = []
    for batch_idx, (img_batch, gt_batch) in enumerate(train_loader, start=1):
        img_batch.to(cfg.device)
        gt_batch.to(cfg.device)
        output = model(img_batch)
        loss = loss_fn(output, gt_batch)
        accuracy = utils.get_accuracy(output, gt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_float = loss.item()
        batch_losses.append(loss_float)
        batch_accs.append(accuracy)
    avg_loss = sum(batch_losses) / len(batch_losses)
    avg_acc = sum(batch_accs) / len(batch_accs)
    return avg_loss, avg_acc
