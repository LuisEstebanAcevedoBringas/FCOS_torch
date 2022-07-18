import FCOS_utils as FCOS_utils
import math
import json
import sys

def save_loss(val, array):
    filename = 'C:\Bringas\MISTI\CVPR_LAB\CVPR_Proyects\FCOS\Losses_FCOS_MM_Dataset.json'
    entry1 = str(val)
    # 1. Read file contents
    with open(filename, "r") as file:
        datos = json.load(file)
    # 2. Update json object
    datos[array].append(entry1)
    # 3. Write json file
    with open(filename, "w") as file:
        json.dump(datos, file)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, batch_size):
    model.train()
    metric_logger = FCOS_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', FCOS_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    acum = 0.0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = FCOS_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        acum += loss_value
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    save_loss(acum/batch_size,"Loss")

    return metric_logger