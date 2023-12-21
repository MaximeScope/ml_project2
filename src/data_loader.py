import numpy as np
import os
from PIL import Image

import torch
from torch.nn.functional import conv2d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class AugDataset(Dataset):
    """
        This class loads the training data and applies the data augmentations
        root: path to the root folder of the dataset
        transform: transformation to apply to the images
        aug: dictionary containing the settings for the augmentations
    """
    def __init__(self, root, aug, transform=None, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.root = root
        self.transform = transform
        self.image_folder = os.path.join(root, "training", "images")
        self.gt_folder = os.path.join(root, "training", "groundtruth")
        self.image_filenames = [
            f for f in os.listdir(self.image_folder) if f.endswith(".png")
        ]
        # Dynamically set instance variables based on hydras config
        # for category, values in aug.items():
        #     for key, value in values.items():
        #         setattr(self, f"{category}_{key}", value)


        # # Define Sobel filter kernels
        # self.sobel_x = (
        #     torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        #     .unsqueeze_(0)
        #     .unsqueeze_(0)
        # )
        # self.sobel_y = (
        #     torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        #     .unsqueeze_(0)
        #     .unsqueeze_(0)
        # )
        # self.obs_size = tuple(aug.obs.size)
        # self.num_obs = aug.obs.n_obs

    def __len__(self):
        return len(self.image_filenames * 8)

    def __getitem__(self, idx):
        id = idx // 8
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


        # Rotate image and ground truth by 90 degrees
        # if idx % 8 in [1, 2, 3]:
        #     image = torch.rot90(image, k=idx % 4, dims=(1, 2))
        #     gt = torch.rot90(gt, k=idx % 4, dims=(0, 1))
        
        # Rotate image and ground truth by 45 degrees
        if idx % 8 != 0:
            image_sh0 = image.shape[1]
            padding = int(2*(1/2**0.5 - 1/2)*image.shape[1]//2)
            gt = torch.unsqueeze(gt, 0)

            image = transforms.functional.pad(image, padding, padding_mode='reflect')
            gt = transforms.functional.pad(gt, padding, padding_mode='reflect')

            image = transforms.functional.rotate(image, idx % 8 * 45, expand=True)
            gt = transforms.functional.rotate(gt, idx % 8 * 45, expand=True)

            image = transforms.functional.center_crop(image, image_sh0)
            gt = transforms.functional.center_crop(gt, image_sh0)

            gt = torch.squeeze(gt, 0)

        # Change the brightness of image and ground truth
        # if idx % 8 in [4, 5, 6, 7]:
        #     # Set the maximum brightness and darkness level randomly between 10% and 40%
        #     br_factor = torch.rand((1,)).item() * (self.br_max_b - self.br_min_b) + self.br_min_b
        #     dark_factor = torch.rand((1,)).item() * (self.br_max_d - self.br_min_d) + self.br_min_d
        #     # Set the direction of the brightness change randomly
        #     br_direction = torch.randint(0, 4, (1,)).item()
        #     # get the position of the idx in the br_indices tensor
        #     if idx % 10 == 4:
        #         img_mask = torch.linspace(br_factor, 1 - br_factor, image.shape[2]).view(1, -1, 1)
        #     elif idx % 10 == 5:
        #         img_mask = torch.linspace(br_factor, 1 - br_factor, image.shape[2]).view(1, 1, -1)
        #     elif idx % 10 == 6:
        #         img_mask = torch.linspace(1 - br_factor, br_factor, image.shape[2]).view(1, -1, 1)
        #     elif idx % 10 == 7:
        #         img_mask = torch.linspace(1 - br_factor, br_factor, image.shape[2]).view(1, 1, -1)
        #     image *= img_mask

        # # Adding noise
        # if idx % 10 == 8:
        #     noise_mask = torch.rand_like(torch.zeros(image.shape)) * (self.noise_max - self.noise_min) - self.noise_min
        #     image += noise_mask 
        #     # Clip the values to be in the valid range [0, 1]
        #     image = torch.clamp(image, 0, 1)

        # # Adding contrast
        # if idx % 10 == 9:
        #     # Calculate the mean intensity of the image
        #     ct_factor = torch.rand(1).item() * (self.ct_max - self.ct_min) - self.ct_min
        #     mean_intensity = image.mean()
        #     # Adjust the image contrast
        #     image = (image - mean_intensity) * (1 + ct_factor) + mean_intensity
        #     # Clip the values to be in the valid range [0, 1]
        #     image = torch.clamp(image, 0, 1)

        # Adding road obstacles
        # if idx in self.obs_indices:
        #     grad_x = conv2d(torch.unsqueeze(gt, 0), self.sobel_x)
        #     grad_y = conv2d(torch.unsqueeze(gt, 0), self.sobel_y)
        #     sobel = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).squeeze_(0)

        #     # get indexes of non-zero elements
        #     ids = torch.nonzero(sobel)

        #     # pick num_obs random indexes
        #     obs_ids = ids[torch.randint(0, len(ids), (self.num_obs,))]

        #     # pick a random color 
        #     color = np.random.uniform(0, 1, (3,))

        #     # add the obstacles to the image
        #     for channel in range(3):
        #         for i in range(-self.obs_size[0] // 2, self.obs_size[0] // 2 + 1):
        #             for j in range(-self.obs_size[1] // 2, self.obs_size[1] // 2 + 1):
        #                 for idx in obs_ids:
        #                     # check if the index is within the image
        #                     if idx[0] + i < image.shape[1] and idx[1] + j < image.shape[2]:
        #                         image[channel, idx[0] + i, idx[1] + j] = color[channel]
        return image, gt

def get_loader(cfg):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = AugDataset(root=cfg.data_path, aug=cfg.augmentation, transform=transform)

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
    data_root = os.path.join(cfg.data_path, 'test_set_images')
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = test_data_loader(root=data_root, transform=transform)

    # Create DataLoader for testing set
    test_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    return test_loader
