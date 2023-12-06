import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import utils

"""
Cross validation function taken from the validation function
in exercise 8. Although, 20% of the training set is used for
validation, instead of using a testing-validating set.
"""


@torch.no_grad()
def test_model(model, device, test_loader, loss_fn):
    model.eval()
    predictions = []
    gts = []
    print("Testing...")
    for img_batch, gt_batch in tqdm(test_loader):
        img_batch, gt_batch = img_batch.to(device), gt_batch.to(device)
        pred = model(img_batch)

        predictions.extend(pred)
        gts.extend(gt_batch)
    predictions = torch.stack(predictions)
    gts = torch.stack(gts)
    avg_loss = loss_fn(predictions, gts)
    avg_f1 = utils.get_f1(predictions, gts)
    #print(f"Test set: Average loss: {avg_loss:0.2e}, f1: {avg_f1:0.3f}")
    return avg_loss, avg_f1


# @torch.no_grad()
# def crossValidate(model, device, train_loader, loss_fn):
#     model.eval()  # Important: eval mode (affects dropout, batch norm etc)
#     val_loss = 0
#     correct = 0

#     # Define the sizes for training and testing sets
#     rest_size = int(0.8 * len(train_loader))
#     val_size = len(train_loader) - rest_size

#     # Split the training set
#     _, val_loader = random_split(train_loader, [rest_size, val_size])

#     for img_batch, gt_batch in val_loader:
#         img_batch, gt_batch = img_batch.to(device), gt_batch.to(device)
#         output = model(img_batch)
#         val_loss += loss_fn(output, gt_batch).item() * len(img_batch)
#         pred = output.argmax(
#             dim=1, keepdim=True
#         )  # get the index of the max log-probability
#         correct += pred.eq(gt_batch.view_as(pred)).sum().item()

#     val_loss /= len(val_loader.dataset)

#     print(
#         "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
#             val_loss,
#             correct,
#             len(val_loader.dataset),
#             100.0 * correct / len(val_loader.dataset),
#         )
#     )
#     return val_loss, correct / len(val_loader.dataset)
