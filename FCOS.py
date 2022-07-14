from torchvision.models.detection.fcos import FCOSClassificationHead #Solo esta disponible en python >= 3.8
import torchvision
import torch
import math

#------- FCOS TransferLearning -------
def FCOS_TransferLearning(num_classes):
    
    #Returns a model pre-trained on COCO train2017
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    
    #Congelar todos los pesos
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    print(model)
    return model

#------- FCOS FineTuning -------
def FCOS_FineTuning(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits
    
    print(model)
    return model

#------- FCOS FromScratch -------
def FCOS_FromScratch(num_classes):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits
    
    print(model)
    return model