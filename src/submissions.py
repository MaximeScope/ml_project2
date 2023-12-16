import os.path

import torch
import torchvision.transforms as T
import PIL
from src import utils


@torch.no_grad()
def get_predictions(model, test_loader, cfg, num=None):
    model.eval()
    predictions = []
    for data in test_loader:
        data = data.to(cfg.device)
        pred_batch = model(data)

        predictions.extend(pred_batch)

        if num is not None and len(predictions) > num:
            break

    return predictions


def save_prediction_masks(predictions, test_loader, cfg):
    img_filenames = []
    for i, prediction in enumerate(predictions):
        img_index = test_loader.dataset.image_indices[i]
        transform = T.ToPILImage()
        mask = transform(prediction)
        if not os.path.exists(cfg.submission_path):
            os.makedirs(cfg.submission_path)
        img_filename = "mask_" + str(img_index) + ".png"
        img_filenames.append(img_filename)
        mask.save(os.path.join(cfg.submission_path, img_filename))

    return img_filenames



# def make_submission(predictions, cfg):
#     output = "id,prediction\n"
#     patched_preds = []
#     for idx, pred_batch in enumerate(predictions):
#         patched_pred = utils.smaller_image(pred_batch, 16)
#         curr_y = 0
#         for i, line in enumerate(patched_pred):
#             curr_x = 0
#             for j, point in enumerate(line):
#                 point_pred = 1 if point > 0.5 else 0
#                 patched_pred[i][j] = point_pred
#                 output += f"{idx:03d}_{curr_y}_{curr_x},{point_pred}\n"
#                 curr_x += 16
#             curr_y += 16
#         patched_pred = utils.bigger_image(patched_pred, cfg, 16)
#         patched_preds.append(patched_pred)
#
#     with open("submission.csv", "w") as f:
#         f.write(output)
#
#     return patched_preds
