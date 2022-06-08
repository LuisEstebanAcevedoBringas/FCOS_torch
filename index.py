from FCOS import  FCOS_TransferLearning, FCOS_FineTuning, FCOS_FromScratch
from create_dataset import VOCDataset, PascalVOCDataset, PennFudanDataset2
from preferences.detect.engine import train_one_epoch, evaluate
from preferences.detect.utils import collate_fn
from transformation import get_transform
from fileinput import filename
import torch
import json
import time

def save_model(epoch, model, optim, name = None):
    if name is None:
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/TransferLearning/prueba_TransferLearning.pth.rar'
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FineTuning/prueba_FineTuning.pth.rar'
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FromScratch/prueba_FromScratch.pth.rar'
    else:
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/TransferLearning/{}.pth.rar'.format(name)
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FineTuning/{}.pth.rar'.format(name)
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FromScratch/{}.pth.rar'.format(name)
    
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optim,
        }

    torch.save(state, filename)

array = ["Losses_TransferLearning"]

filename = './results/Losses_TransferLearning.json'
with open(filename, "r") as file:
    datos = json.load(file)
for i in array:
    datos[i].clear()
with open(filename, "w") as file:
    json.dump(datos, file)

#Hiperparametros
lr = 1e-4
momentum = 0.9
weight_decay = 8e-4

def main(checkpoint = None):
    
    global lr, momentum, weight_decay, start_epoch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21

    dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TRAIN', get_transform(True))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    if checkpoint is None:
        
        start_epoch = 0
        model = FCOS_TransferLearning(num_classes) #LLamamos la funcion para cambiar el metodo de entrenamiento a TL
        #model = FCOS_FineTuning(num_classes) #LLamamos la funcion para cambiar el metodo de entrenamiento a FT
        #model = FCOS_FromScratch(num_classes) #LLamamos la funcion para cambiar el metodo de entrenamiento a FS
        model.to(device)

        biases = []
        not_biases = []

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], 
                                    lr=lr,
                                    momentum=momentum, 
                                    weight_decay=weight_decay)
    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # let's train it for 10 epochs
    print(start_epoch)
    num_epochs = 232
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    frontera = 40

    tiempo_entrenamiento = time.time()
    for epoch in range(start_epoch ,num_epochs):
        # train for one epoch, printing every 200 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        # update the learning rate
        if not epoch < frontera:
            lr_scheduler.step()

        if epoch == 39:
            save_model(epoch, model, optimizer, "prueba_39_f")

        if epoch == 60:
            save_model(epoch, model, optimizer, "prueba_60_f")
        
        if epoch == 80:
            save_model(epoch, model, optimizer, "prueba_80_f")

        if epoch == 100:
            save_model(epoch, model, optimizer, "prueba_100_f")

        if epoch == 130:
            save_model(epoch, model, optimizer, "prueba_130_f")

        if epoch == 160:
            save_model(epoch, model, optimizer, "prueba_160_f")
        #evaluate on the test dataset
        #evaluate(model, data_loader, device=device)
        save_model(epoch, model, optimizer)

    tiempo_entrenamiento = time.time() - tiempo_entrenamiento
    print("That's it!")

    print("Tiempo_entrenamiento: ", tiempo_entrenamiento)

main('/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/prueba_TransferLearning.pth.rar')