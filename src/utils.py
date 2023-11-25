import torch

# Make bigger pixels of size patch_size x patch_size
def smaller_image(img, patch_size):
    """
    img is a torch tensor of shape (3, 400, 400)
    devide the 400 x 400 image into 400/patch_size x 400/patch_size pixels
    by averaging the rgb values for each pixel.
    """
    if len(img.shape) == 2:
        img = img.reshape(
            int(img.shape[0]/patch_size),
            patch_size,
            int(img.shape[1]/patch_size),
            patch_size
        ).mean(3).mean(1)
    else:
        img = img.reshape(
            img.shape[0],
            int(img.shape[1]/patch_size),
            patch_size,
            int(img.shape[2]/patch_size),
            patch_size
        ).mean(4).mean(2)

    return img

def bigger_image(img, patch_size):

    # Reshape the tensor using kron:
    kron_param = torch.ones(patch_size, patch_size, dtype=img.dtype)
    scaled_image = torch.kron(img, kron_param)

    return scaled_image
