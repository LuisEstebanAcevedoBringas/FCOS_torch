from torchvision.models.detection.fcos import FCOSClassificationHead # Esta chingadera no esta descargada
from torchvision.models.detection import _utils as det_utils
from functools import partial
from pickle import TRUE
from torch import nn 
import torchvision

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
    
    model.head.classification_head = FCOSClassificationHead(num_classes)

    print(model)
    return model

#FCOS FineTuning
def FCOS_FineTuning(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False)
    
    model.head.classification_head = FCOSClassificationHead(num_classes)

    print(model)
    return model

#FCOS FromScratch
def FCOS_FromScratch(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False)
    
    model.head.classification_head = FCOSClassificationHead(num_classes)

    print(model)
    return model

