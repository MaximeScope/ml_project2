import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

"""
Cross validation function taken from the validation function
in exercise 8. Although, 20% of the training set is used for
validation, instead of using a testing-validating set.
"""

@torch.no_grad()
def crossValidate(model, device, train_loader, loss_fn):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    val_loss = 0
    correct = 0
    
    # Define the sizes for training and testing sets
    rest_size = int(0.8 * len(train_loader))
    val_size = len(train_loader) - rest_size

    # Split the training set
    _, val_loader = random_split(train_loader, [rest_size, val_size])

    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += loss_fn(output, target).item() * len(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            val_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return val_loss, correct / len(val_loader.dataset)

@torch.no_grad()
def get_predictions(model, device, test_loader, loss_fn, num=None):
    model.eval()
    points = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points