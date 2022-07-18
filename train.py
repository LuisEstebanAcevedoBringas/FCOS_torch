from FCOS import  FCOS_TransferLearning, FCOS_FineTuning, FCOS_FromScratch
from create_dataset import VOCDataset,Dataset_G
from transformation import get_transform
from engine import train_one_epoch
from FCOS_utils import collate_fn
import torch
import time

def save_model(epoch, model, optim, name = None):
    if name is None:
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/MM_Dataset/Checkpoint_FineTuning_MM_Dataset.pth.rar'
    else:
        filename = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/MM_Dataset/{}.pth.rar'.format(name)
    
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
decay_lr_at = [20 , 25]
batch_size = 4

def main(checkpoint = None):
    
    global lr, momentum, weight_decay, start_epoch, decay_lr_at

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 6 #Mask Dataset
    #num_classes = 21 #PascalVOC

    dataset = Dataset_G('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles_G', 'TRAIN', get_transform(True)) #Mask Dataset
    #dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TRAIN', get_transform(True)) #PascalVOC
    
    data_loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    if checkpoint is None:
        start_epoch = 1
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

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 1 * lr}, {'params': not_biases}], 
            lr=lr,
            momentum=momentum, 
            weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Empieza el entrenamiento
    print(start_epoch)

    num_epochs = 30
    
    tiempo_entrenamiento = time.time()

    for epoch in range(start_epoch ,num_epochs):

        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, 0.1)

        train_one_epoch(model, optimizer, data_loader, device, epoch,batch_size= len(data_loader), print_freq = 200)

        if epoch == 5:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_05")

        if epoch == 10:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_10")

        if epoch == 15:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_15")

        if epoch == 20:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_20")

        if epoch == 21:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_21")

        if epoch == 24:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_24")

        if epoch == 25:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_25")

        if epoch == 26:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_26")

        if epoch == 27:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_27")

        if epoch == 28:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_28")

        if epoch == 29:
            save_model(epoch, model, optimizer, "Checkpoint_FT_MM_Dataset_epoca_29")

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

main() #Si existe un checkpoint, se debe de agregar la ruta aqui.