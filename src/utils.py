# Make bigger pixels of size batch_size x batch_size
def smaller_image(img, batch_size):
    """
    img is a torch tensor of shape (3, 400, 400)
    devide the 400 x 400 image into 400/batch_size x 400/batch_size pixels
    by averaging the rgb values for each pixel.
    """
    img = img.reshape(
        3,
        int(img.shape[1]/batch_size),
        batch_size,
        int(img.shape[2]/batch_size),
        batch_size
    ).mean(4).mean(2)

    return img
