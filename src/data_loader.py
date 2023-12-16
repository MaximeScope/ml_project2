import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


"""
    This function loads the training data and applies the specified transformations.
    root: path to the root folder of the dataset
    transform: transformation to apply to the images
    aug: dictionary containing the settings for the augmentations
"""
def torch_loader(root, transform, aug):
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
            
            # Define the added transformations (brightness, noise and contrast)
            self.mod_indices = torch.randperm(len(self.image_filenames * 4))
            # Pick 40 images for brightness change
            self.br_indices = self.mod_indices[:aug.br.n_img]
            self.noise_indices = self.mod_indices[
                aug.br.n_img:aug.noise.n_img + aug.br.n_img
            ]
            self.ct_indices = self.mod_indices[
                aug.noise.n_img + aug.br.n_img:aug.ct.n_img + aug.noise.n_img + aug.br.n_img
            ]
            self.br_factor = []
            self.dark_factor = []
            self.br_direction = []
            for i in range(len(self.br_indices)):
                # Set the maximum brightness and darkness level randomly between 10% and 40%
                self.br_factor.append(torch.rand((1,)).item() * (aug.br.max_b - aug.br.min_b) + aug.br.min_b)
                self.dark_factor.append(torch.rand((1,)).item() * (aug.br.max_d - aug.br.min_d) + aug.br.min_d)
                # Set the direction of the brightness change randomly
                self.br_direction.append(torch.randint(0, 4, (1,)).item())
            self.noise_mask = []
            for i in range(len(self.noise_indices)):
                # Generate a noise mask with values between -10% and 10% for each channel
                img_name = os.path.join(self.image_folder, self.image_filenames[i])
                image = Image.open(img_name).convert("RGB")
                image = self.transform(image)
                self.noise_mask.append(torch.rand_like(torch.zeros(image.shape)) * (aug.noise.max - aug.noise.min) - aug.noise.min)
            self.ct_factor = []
            for i in range(len(self.ct_indices)):
                # Set the contrast level randomly between -30% and 30%
                self.ct_factor.append(torch.rand(1).item() * (aug.ct.max - aug.ct.min) - aug.ct.min)

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

            if idx % 4 == 1 or idx % 4 == 2 or idx % 4 == 3:
                # Rotate image and ground truth by 90 degrees
                image = torch.rot90(image, k=idx % 4, dims=(1, 2))
                gt = torch.rot90(gt, k=idx % 4, dims=(0, 1))
            
            # Change the brightness of image and ground truth of 40 samples
            if idx in self.br_indices:
                # get the position of the idx in the br_indices tensor
                i = (self.br_indices == idx).nonzero(as_tuple=True)[0]
                if self.br_direction[i] == 0:
                    img_mask = torch.linspace(1 + self.br_factor[i], 1 - self.dark_factor[i], image.shape[2]).view(1, -1, 1)
                elif self.br_direction[i] == 1:
                    img_mask = torch.linspace(1 + self.br_factor[i], 1 - self.dark_factor[i], image.shape[2]).view(1, 1, -1)
                elif self.br_direction[i] == 2:
                    img_mask = torch.linspace(1 - self.dark_factor[i], 1 + self.br_factor[i], image.shape[2]).view(1, -1, 1)
                elif self.br_direction[i] == 3:
                    img_mask = torch.linspace(1 - self.dark_factor[i], 1 + self.br_factor[i], image.shape[2]).view(1, 1, -1)
                image *= img_mask

            # Adding noise
            if idx in self.noise_indices:
                # get the position of the idx in the noise_indices tensor
                i = (self.noise_indices == idx).nonzero(as_tuple=True)[0]
                # Apply the noise mask to each channel independently
                image += self.noise_mask[i]
                # Clip the values to be in the valid range [0, 1]
                image = torch.clamp(image, 0, 1)

            # Adding contrast
            if idx in self.ct_indices:
                # get the position of the idx in the ct_indices tensor
                i = (self.ct_indices == idx).nonzero(as_tuple=True)[0]
                # Calculate the mean intensity of the image
                mean_intensity = image.mean()
                # Adjust the image contrast
                image = (image - mean_intensity) * (1 + self.ct_factor[i]) + mean_intensity
                # Clip the values to be in the valid range [0, 1]
                image = torch.clamp(image, 0, 1)

            return image, gt

    dataset = TheDataset(root=root, transform=transform)

    return dataset


def get_loader(cfg):
    data_root = cfg.data_path
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = torch_loader(root=data_root, transform=transform, aug=cfg.augmentation)

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
