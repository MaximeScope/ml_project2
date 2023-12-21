import os.path

import torch
import torchvision.transforms as T


@torch.no_grad()
def get_predictions(model, test_loader, cfg, num=None):
    model.eval()
    predictions = []
    # Iterate over the test set image and get predictions with the trained model
    for data in test_loader:
        data = data.to(cfg.device)
        pred_batch = model(data)

        predictions.extend(pred_batch)

        if num is not None and len(predictions) > num:
            break

    return predictions


def save_prediction_masks(predictions, test_loader, cfg):
    img_filenames = []
    # Iterate over the predictions and save them as a PNG grayscale mask
    for i, prediction in enumerate(predictions):
        # Get the correct index for the image
        img_index = test_loader.dataset.image_indices[i]
        # Convert the prediction data to a PNG image
        transform = T.ToPILImage()
        mask = transform(prediction)
        if not os.path.exists(cfg.submission_path):
            os.makedirs(cfg.submission_path)
        img_filename = "mask_" + str(img_index) + ".png"
        img_filenames.append(img_filename)
        mask.save(os.path.join(cfg.submission_path, img_filename))

    return img_filenames
