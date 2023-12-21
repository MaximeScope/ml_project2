# Team LuAnMa CS-433 ML Project 2 - Road Segmentation

This project runs our best performing model for the road segmentation project. The best F1 score obtained was 0.897

## Dataset
We used the provided set of satellite images acquired from GoogleMaps and the ground-truth images where each pixel is labeled 
as road or background.

## Data Augmentation
We generated every possible 45-degree rotation of each picture in the training data to augment the data set. This is the
only data augmentation we use in our final submission

## Model

Our model uses a U-Net with a depth of 16, a learning rate of $1 \times 10^{-3}$ and a weight
decay of $1 \times 10^{-4}$. The best F1 score was obtained by running our model over 500 epochs.
The U-Net implementation used in the project was obtained from https://github.com/milesial/Pytorch-UNet/tree/master

## Running the code
`python3 -m run`