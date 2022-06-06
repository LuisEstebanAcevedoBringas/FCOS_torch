from cv2 import imshow
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.ops import nms

plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from torchvision.transforms.functional import convert_image_dtype
import glob
import random
import json
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

filename = '/home/bringascastle/Documentos/repos/SSD/results/losses.json'

# lab = ['NMS', 'general', 'draw', 'red']

# with open(filename, "r") as file:
#     datos = json.load(file)
#     # 2. Update json object
# for i in lab:
#     datos[i].clear()
#     # 3. Write json file
# with open(filename, "w") as file:
#     json.dump(datos, file)


resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#Get time TransferLearning
def get_times(val, array):
    filename = '/home/bringascastle/Documentos/repos/SSD/results/losses.json'
    entry1 = str(val)
    # 1. Read file contents
    with open(filename, "r") as file:
        datos = json.load(file)
    # 2. Update json object
    datos[array].append(entry1)
    # 3. Write json file
    with open(filename, "w") as file:
        json.dump(datos, file)

chk = torch.load('/home/bringascastle/Documentos/repos/SSD/bin/prueba_t_10.pth.rar')

star = chk['epoch'] + 1
print('Ultima epoca de entrenamiento: {}'.format(star))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
model = chk['model']
model.eval()
model.to(device)

list_r = list(range(0, 4952))
random.shuffle(list_r)
list_cut = list_r[0:32]
list_path = glob.glob('/home/bringascastle/Escritorio/datasets/VOC_datasets/test_2007/JPEGImages/*.jpg')

for i in list_cut:
    image_prob = read_image(list_path[i])
    batch = torch.stack([image_prob.to(device)])
    batch = convert_image_dtype(batch, dtype=torch.float)
    batch.to(device)

    times_general = time.time()
    times = time.time()
    output = model(batch)
    #get_times(time.time() - times, "red")
    score_threshold = .45

    bbox = output[0]['boxes']
    
    a = []
    lista = []
    listb = []
    for i in range(len(bbox)):
        if output[0]['scores'][i] > score_threshold:
            val = output[0]['labels'][i] - 1

            lista.append(voc_labels[val])
            listb.append(distinct_colors[val])

            a.append(bbox[i].tolist())

    a = torch.tensor(a, dtype=torch.float)
    times = time.time()
    # draw bounding box on the input image
    img = draw_bounding_boxes(image_prob, a , width=3 ,labels=lista,colors=listb)

    #get_times(time.time()- times, 'draw')

    #get_times(time.time() - times_general, 'general')

    img = torchvision.transforms.ToPILImage()(img)
    img.show()