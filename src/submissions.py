import torch
from src import utils


@torch.no_grad()
def get_predictions(model, test_loader, cfg, num=None):
    model.eval()
    predictions = []
    for data, _ in test_loader:
        data = data.to(cfg.device)
        pred_batch = model(data)

        predictions.extend(pred_batch)

        if num is not None and len(predictions) > num:
            break

    return predictions


def make_submission(predictions, cfg):
    output = "id,prediction\n"
    patched_preds = []
    for idx, pred_batch in enumerate(predictions):
        patched_pred = utils.smaller_image(pred_batch, 16)
        curr_y = 0
        for i, line in enumerate(patched_pred):
            curr_x = 0
            for j, point in enumerate(line):
                point_pred = 1 if point > 0.5 else 0
                patched_pred[i][j] = point_pred
                output += f"{idx:03d}_{curr_y}_{curr_x},{point_pred}\n"
                curr_x += 16
            curr_y += 16
        patched_pred = utils.bigger_image(patched_pred, cfg, 16)
        patched_preds.append(patched_pred)

    with open("submission.csv", "w") as f:
        f.write(output)

    return patched_preds
