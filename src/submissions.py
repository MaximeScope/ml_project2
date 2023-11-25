import torch
import utils


@torch.no_grad()
def get_predictions(model, device, val_loader, num=None):
    model.eval()
    points = []
    for data, idxs in val_loader:
        data = data.to(device)
        pred = model(data)

        points.extend(zip(idxs, pred))

        if num is not None and len(points) > num:
            break

    return points


def make_submission(points):
    output = "id,prediction\n"
    for data_tuple in points:
        for idx, pred in data_tuple:
            patched_pred = utils.smaller_image(pred, 16)
            curr_y = 0
            for line in patched_pred:
                curr_x = 0
                for point in line:
                    point_pred = 1 if point > 0.5 else 0
                    output += f"{idx:03d}_{curr_y}_{curr_x},{point_pred}\n"
                    curr_x += 16
                curr_y += 16

    with open("../submission.csv", "w") as f:
        f.write(output)