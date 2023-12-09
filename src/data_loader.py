import numpy as np
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
            self.image_folder = os.path.join(root, "training", "images")
            self.gt_folder = os.path.join(root, "training", "groundtruth")
            self.image_filenames = [
                f for f in os.listdir(self.image_folder) if f.endswith(".png")
            ]

        def __len__(self):
            return len(self.image_filenames * 4)

        def __getitem__(self, idx):
            id = idx // 4
            img_name = os.path.join(self.image_folder, self.image_filenames[id])
            gt_name = os.path.join(
                self.gt_folder, self.image_filenames[id]
            )  # gt standing for groundtruth

            image = Image.open(img_name).convert("RGB")
            gt = Image.open(gt_name).convert("L")  # L mode represents grayscale

            if self.transform:
                image = self.transform(image)

            # Convert ground truth to binary mask (streets in white, everything else in black)
            gt = torch.tensor(np.array(gt) > 128, dtype=torch.float32)

            if idx % 4 != 0:
                image = torch.rot90(image, k=idx % 4, dims=(1, 2))
                gt = torch.rot90(gt, k=idx % 4, dims=(0, 1))

            return image, gt

    dataset = TheDataset(root=root, transform=transform)

    return dataset


def get_loader(cfg):
    data_root = cfg.data_path
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torch_loader(root=data_root, transform=transform)

    # Define the sizes for training and testing sets
    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size

    # Split the dataset
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training set
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )

    # # Create DataLoader for testing set
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=cfg.training.batch_size,
    #     shuffle=False,
    # )

    return train_loader


def test_data_loader(root, transform):
    class TheDataset(Dataset):
        def __init__(self, root, transform=None, *args, **kwargs):
            super(TheDataset, self).__init__(*args, **kwargs)
            self.root = root
            self.transform = transform
            self.image_indices = [int(f.split("_")[1]) for f in os.listdir(self.root)]
            self.image_folder = os.path.join(root, "test_set_images")

        def __len__(self):
            return len(self.image_indices)

        def __getitem__(self, idx):
            # Make sure we get the correct index for the correct image
            img_name = os.path.join(
                self.root,
                "test_" + str(self.image_indices[idx]),
                "test_" + str(self.image_indices[idx]) + ".png",
            )

            image = Image.open(img_name).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image

    dataset = TheDataset(root=root, transform=transform)

    return dataset


def get_test_loader(cfg):
    data_root = cfg.test_data_path
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = test_data_loader(root=data_root, transform=transform)

    batch_size = 32
    # Create DataLoader for testing set
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_loader
