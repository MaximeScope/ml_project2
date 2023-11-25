def train_model(model, optimizer, loss_fn, train_loader, val_loader, cfg):
    train_losses = []
    train_accs = []
    for epoch in range(1, cfg.training.epochs + 1):
        batch_losses, batch_accs = train_epoch(
            model, optimizer, loss_fn, train_loader, epoch, cfg
        )
        train_losses.extend(batch_losses)
        train_accs.extend(batch_accs)
    return train_losses, train_accs

def train_epoch(model, optimizer, loss_fn, train_loader, epoch, cfg):
    model.train()
    batch_losses = []
    batch_accs = []
    for batch_idx, (img_batch, gt_batch) in enumerate(train_loader):
        img_batch.to(cfg.device)
        gt_batch.to(cfg.device)
        output = model(img_batch)
        loss = loss_fn(output, gt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: determine accuracy
        loss_float = loss.item()
        batch_losses.append(loss_float)
        batch_accs.append(accuracy_float)
        if batch_idx % (len(train_loader.dataset) // len(img_batch) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
            )
    return batch_losses, batch_accs