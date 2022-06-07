from torchvision.models.detection.fcos import FCOSClassificationHead # Esta chingadera no esta descargada
from torchvision.models.detection import _utils as det_utils
from functools import partial
from pickle import TRUE
from torch import nn 
import torchvision
import torch
import math

#Args:
    #in_channels (int): number of channels of the input feature.
    #num_anchors (int): number of anchors to be predicted.
    #num_classes (int): number of classes to be predicted.
    #num_convs (Optional[int]): number of conv layer. Default: 4.
    #prior_probability (Optional[float]): probability of prior. Default: 0.01.
    #norm_layer: Module specifying the normalization layer to use.

#FCOS TransferLearning
def FCOS_TransferLearning(num_classes):
    
    #Returns a model pre-trained on COCO train2017
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    
    #Congelar todos los pesos
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    out_channels = 256

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    print(model)
    return model

#FCOS FineTuning
def FCOS_FineTuning(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    out_channels = 256

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    print(model)
    return model

#FCOS FromScratch
def FCOS_FromScratch(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    out_channels = 256

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    print(model)
    return model

