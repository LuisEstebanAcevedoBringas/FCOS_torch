from FCOS import  FCOS_TransferLearning, FCOS_FineTuning, FCOS_FromScratch
from preferences.detect.engine import train_one_epoch
from create_dataset import VOCDataset,Dataset_G
from preferences.detect.utils import collate_fn
from transformation import get_transform
import torch
import time

def save_model(epoch, model, optim, name = None):
    if name is None:
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/TransferLearning/prueba_TransferLearning.pth.rar'
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FineTuning/Checkpoint_FineTuning_G.pth.rar'
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FromScratch/prueba_FromScratch.pth.rar'
    else:
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/TransferLearning/{}.pth.rar'.format(name)
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FineTuning/{}.pth.rar'.format(name)
        #filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FromScratch/{}.pth.rar'.format(name)
    
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optim,
        }

    torch.save(state, filename)

# array = ["Losses_TransferLearning"]

# filename = './results/Losses_TransferLearning.json'
# with open(filename, "r") as file:
#     datos = json.load(file)
# for i in array:
#     datos[i].clear()
# with open(filename, "w") as file:
#     json.dump(datos, file)

#Hiperparametros
lr = 1e-4
momentum = 0.9
weight_decay = 5e-4
iterations = 120000
decay_lr_at = [80000, 100000]

def main(checkpoint = None):
    
    global lr, momentum, weight_decay, start_epoch, decay_lr_at

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    num_classes = 6

    dataset = Dataset_G('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles_G', 'TRAIN', get_transform(True))
    #dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TRAIN', get_transform(True))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    if checkpoint is None:
        
        start_epoch = 0
        #model = FCOS_TransferLearning(num_classes) #LLamamos la funcion para cambiar el metodo de entrenamiento a TL
        model = FCOS_FineTuning(num_classes) #LLamamos la funcion para cambiar el metodo de entrenamiento a FT
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

    # let's train
    print(start_epoch)

    num_epochs = 232
    
    decay_lr_at = [it // (len(dataset) // 32) for it in decay_lr_at]

    tiempo_entrenamiento = time.time()

    for epoch in range(start_epoch ,num_epochs):

        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, 0.1)

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 200)

        if epoch == 5:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_05")

        if epoch == 10:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_10")

        if epoch == 15:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_15")
        
        if epoch == 20:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_20")
        
        if epoch == 25:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_25")

        if epoch == 30:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_30")

        if epoch == 35:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_35")
        
        if epoch == 40:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_40")
        
        if epoch == 45:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_45")

        if epoch == 50:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_50")

        if epoch == 55:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_55")

        if epoch == 60:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_60")

        if epoch == 100:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_100")

        if epoch == 154:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_154")

        if epoch == 195:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_195")

        if epoch == 215:
            save_model(epoch, model, optimizer, "Checkpoint_G_FT_epoca_215")

        save_model(epoch, model, optimizer)

    tiempo_entrenamiento = time.time() - tiempo_entrenamiento
    print("That's it!")

    print("Tiempo_entrenamiento: ", tiempo_entrenamiento)

def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

main()