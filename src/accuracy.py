import torch

def get_accuracy(predictions, gts):
    """
    Get the accuracy of the predictions compared to the ground truth over the whole batch
    :param predictions: predictions of the model
    :param gts: ground truths
    :return: accuracy
    """
    correct_pixels = torch.eq(predictions, gts).sum().item()
    total_pixels = predictions.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy