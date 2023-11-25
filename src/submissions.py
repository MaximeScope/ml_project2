import torch
import numpy as np


@torch.no_grad()
def get_predictions(model, device, val_loader, num=None):
    model.eval()
    points = []
    for data, idxs in val_loader:
        data = data.to(device)
        pred = model(data)

        pred = np.split(pred.cpu().numpy(), len(data))
        points.extend(zip(idxs, pred))

        if num is not None and len(points) > num:
            break

    return points
