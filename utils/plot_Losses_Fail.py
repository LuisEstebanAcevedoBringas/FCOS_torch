import matplotlib.pyplot as plt
import json

#Get the data from Losses_FromScratch.json file
filename_FS = 'Losses_FromScratch.json'
with open(filename_FS, "r") as file:
    datosFS = json.load(file)

#Get the data from Losses_TransferLearning.json file
filename_TL = 'Losses_TransferLearning.json'
with open(filename_TL, "r") as file:
    datosTL = json.load(file)

#Get the data from Losses_FineTuning.json file
filename_FT = 'Losses_FineTuning.json'
with open(filename_FT, "r") as file:
    datosFT = json.load(file)

def get_lists_for_plt(array):
    aux = []
    for i in range(59, len(array) + 1, 60):
        aux.append(float(array[i]))
    return aux

l1 = get_lists_for_plt(datosFS['Losses_FromScratch'])
l2 = get_lists_for_plt(datosTL['Losses_TransferLearning'])
l3 = get_lists_for_plt(datosFT['Losses_FineTuning'])

epochs = 232

epochs = range(1, epochs + 1, 1)

plt.plot(epochs, l1, 'r', label='Losses  - From Scratch')
plt.plot(epochs, l2, 'b', label='Losses -  Transfer Learning')
plt.plot(epochs, l3, 'g', label='Losses - Finetuning')

plt.title('Losses FCOS')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()
plt.figure()
plt.show()
