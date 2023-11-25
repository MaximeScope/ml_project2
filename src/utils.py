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

# Make bigger pixels of size batch_size x batch_size
def smaller_image(img, batch_size):
    """
    img is a torch tensor of shape (3, 400, 400)
    devide the 400 x 400 image into 400/batch_size x 400/batch_size pixels
    by averaging the rgb values for each pixel.
    """
    if len(img.shape) == 2:
        img = img.reshape(
            int(img.shape[0]/batch_size),
            batch_size,
            int(img.shape[1]/batch_size),
            batch_size
        ).mean(3).mean(1)
    else:
        img = img.reshape(
            img.shape[0],
            int(img.shape[1]/batch_size),
            batch_size,
            int(img.shape[2]/batch_size),
            batch_size
        ).mean(4).mean(2)

    return img

def bigger_image(img, batch_size):

    # Reshape the tensor using kron:
    kron_param = torch.ones(batch_size, batch_size, dtype=img.dtype)
    scaled_image = torch.kron(img, kron_param)

    return scaled_image
