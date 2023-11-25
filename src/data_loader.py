import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def torch_loader(root, transform):

    class TheDataset(Dataset):
        def __init__(self, root, transform=None, *args, **kwargs):
            super(TheDataset, self).__init__(*args, **kwargs)
            self.root = root
            self.transform = transform
            self.image_folder = os.path.join(root, 'training', 'images')
            self.gt_folder = os.path.join(root, 'training', 'groundtruth')
            self.image_filenames = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_folder, self.image_filenames[idx])
            gt_name = os.path.join(self.gt_folder, self.image_filenames[idx]) # gt standing for groundtruth

            image = Image.open(img_name).convert('RGB')
            gt = Image.open(gt_name).convert('L')  # L mode represents grayscale

            if self.transform:
                image = self.transform(image)

            # Convert ground truth to binary mask (streets in white, everything else in black)
            gt = torch.tensor(np.array(gt) > 128, dtype=torch.float32)

            return image, gt

    dataset = TheDataset(root=root, transform=transform)

    return dataset

def get_loader(cfg):
    data_root = cfg.data_path
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torch_loader(root=data_root, transform=transform)

    # Define the sizes for training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Create DataLoader for testing set
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    return train_loader, test_loader

def plot_random_sample(train_loader):
    # Set the figure size based on the number of samples
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Get the length of the dataset
    dataset_size = len(train_loader.dataset)

    # Generate random indices using torch.randperm
    random_indices = torch.randperm(dataset_size)[:1]

    for idx in range(len(random_indices)):
        # Get the sample using the generated index
        image, groundtruth = train_loader.dataset[idx]

        # Plot the original image
        axes[0].imshow(image.permute(1, 2, 0)) # Permute to (H, W, C) for plotting
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot the ground truth
        axes[1].imshow(groundtruth, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()