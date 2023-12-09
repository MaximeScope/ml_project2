import torch
from torchvision.transforms.functional import rotate, to_pil_image, to_tensor


@torch.no_grad()
def get_f1(predictions, gts):
    """
    Get the f1 of the predictions compared to the ground truth over the whole patch
    :param predictions: predictions of the model
    :param gts: ground truths
    :return: accuracy
    """
    predictions_tresholded = (predictions > 0.5).float()
    # get f1 score
    tp = torch.sum(predictions_tresholded * gts)
    fp = torch.sum(predictions_tresholded * (1 - gts))
    fn = torch.sum((1 - predictions_tresholded) * gts)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


# Make bigger pixels of size patch_size x patch_size
def smaller_image(img, patch_size):
    """
    img is a torch tensor of shape (3, 400, 400)
    devide the 400 x 400 image into 400/patch_size x 400/patch_size pixels
    by averaging the rgb values for each pixel.
    """
    if len(img.shape) == 2:
        img = (
            img.reshape(
                int(img.shape[0] / patch_size),
                patch_size,
                int(img.shape[1] / patch_size),
                patch_size,
            )
            .mean(3)
            .mean(1)
        )
    else:
        img = (
            img.reshape(
                img.shape[0],
                int(img.shape[1] / patch_size),
                patch_size,
                int(img.shape[2] / patch_size),
                patch_size,
            )
            .mean(4)
            .mean(2)
        )

    return img


def bigger_image(img, cfg, patch_size):
    # Reshape the tensor using kron:
    kron_param = torch.ones(patch_size, patch_size, dtype=img.dtype, device=cfg.device)
    scaled_image = torch.kron(img, kron_param)

    return scaled_image

# def rotate_batch(image, amount, dims):
#     """
#     Expends the dataset by rotating the images and the groundtruths
#     """

#     # rotation_matrices = [
#     #     torch.tensor([[0, -1], [1, 0]]),
#     #     torch.tensor([[-1, 0], [0, -1]]),
#     #     torch.tensor([[0, 1], [-1, 0]])
#     # ]

#     print(image.shape)

#     image_rotated = torch.rot90(image, k=amount, dims=dims)

#     return image_rotated