import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
import random
import torch

from src import utils


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

        if len(indices) == 1:
            # Plot the original image
            axes[0].imshow(image.permute(1, 2, 0))  # Permute to (H, W, C) for plotting
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            # Plot the ground truth
            axes[1].imshow(groundtruth, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
        else:
            axes[i, 0].imshow(
                image.permute(1, 2, 0)
            )  # Permute to (H, W, C) for plotting
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")
            # Plot the ground truth
            axes[i, 1].imshow(groundtruth, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# Display prediction as an image
def plot_prediction(test_loader, pred, indice):
    # Set the figure size based on the number of samples
    _, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Get the sample using the generated index
    image, groundtruth = test_loader.dataset[indice]

    # Plot the original image
    axes[0].imshow(image.permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    # Plot the ground truth
    axes[1].imshow(groundtruth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    # Plot the prediction
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_pred_on(test_loader, predictions, indice, cfg):
    # Set the figure size based on the number of samples
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Get the prediction
    pred = predictions[indice]

    # Get the sample using the generated index
    image = test_loader.dataset[indice]

    # Scale the prediction to the size of the image
    if pred.shape[0] != image.shape[1] or pred.shape[1] != image.shape[2]:
        pred = utils.bigger_image(
            pred, cfg, patch_size=int(image.shape[1] / pred.shape[0])
        )

    # Create a red-to-white colormap
    cmap = LSC.from_list("red_to_white", ["red", "white"])

    # # Plot the original image
    # axes[0].imshow(image.permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    # axes[0].imshow(groundtruth, cmap=cmap, alpha=0.5)
    # axes[0].set_title("Ground Truth")
    # axes[0].axis("off")
    # Plot the ground truth
    axes[1].imshow(image.permute(1, 2, 0))  # Permute to (H, W, C) for plotting
    axes[1].imshow(pred, cmap=cmap, alpha=0.5)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_train(train_losses, train_f1s):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(train_losses, color='k')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(train_f1s, color="red")
    ax2.set_ylabel("F1", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    plt.show()
