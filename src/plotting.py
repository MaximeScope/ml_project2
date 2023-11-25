import matplotlib.pyplot as plt
import random
import torch

import utils

def plot_random_sample(train_loader, indices=None):

    # Get the length of the dataset
    dataset_size = len(train_loader.dataset)

    if indices is None:
        # Set a random amount of indices to plot
        n_samples = random.randint(1, 4)
        # Generate random indices
        indices = random.sample(range(dataset_size), n_samples)
    
    # Set the figure size based on the number of samples
    _, axes = plt.subplots(len(indices), 2, figsize=(8, 4))

    for i, idx in enumerate(indices):
        # Get the sample using the generated index
        image, groundtruth = train_loader.dataset[idx]

        image = utils.smaller_image(image, batch_size=16)

        groundtruth = utils.smaller_image(groundtruth, batch_size=16)

        if len(indices) == 1:
            # Plot the original image
            axes[0].imshow(image.permute(1, 2, 0)) # Permute to (H, W, C) for plotting
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            # Plot the ground truth
            axes[1].imshow(groundtruth, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        else:
            axes[i, 0].imshow(image.permute(1, 2, 0)) # Permute to (H, W, C) for plotting
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            # Plot the ground truth
            axes[i, 1].imshow(groundtruth, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Display prediction as an image
def plot_prediction(img, gt, prediction):
    # Set the figure size
    _, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the original image
    axes[0].imshow(img.permute(1, 2, 0)) # Permute to (H, W, C) for plotting
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the ground truth
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Plot the prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_pred_on(img, gt, prediction):
    w = gt_imgs[img_idx].shape[0]
    h = gt_imgs[img_idx].shape[1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
    cimg = concatenate_images(imgs[img_idx], predicted_im)
    fig1 = plt.figure(figsize=(10, 10))  # create a figure with the default size
    plt.imshow(cimg, cmap="Greys_r")

    new_img = make_img_overlay(imgs[img_idx], predicted_im)

    plt.imshow(new_img)