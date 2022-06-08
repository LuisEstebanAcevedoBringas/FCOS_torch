import matplotlib.pyplot as plt
from calendar import EPOCH
import json

epoch = 232

filename2 = '/home/fp/Escritorio/LuisBringas/FCOS/results/Losses_TransferLearning.json' #TransferLearning
filename1 = '/home/fp/Escritorio/LuisBringas/FCOS/results/Losses_FineTuning.json' #FineTuning
filename3 = '/home/fp/Escritorio/LuisBringas/FCOS/results/Losses_FromScratch.json' #FromScratch

with open(filename1, "r") as file:
    finetunning = json.load(file)
    # 2. Update json object
with open(filename2, "r") as file:
    transfer = json.load(file)
    # 2. Update json object
with open(filename3, "r") as file:
    fromscratch = json.load(file)
    # 2. Update json object

def getArray(data):
    a = []
    for i in range(0, len(data), 2069):
        a.append(float(data[i]))
    return a

epoch = range(1, epoch , 1)

a, b, c = getArray(finetunning['Losses_FineTuning']), getArray(transfer['Losses_TransferLearning']), getArray(fromscratch['Losses_FromScratch'])

epoch = range(1, len(a) + 1, 1)
plt.plot ( epoch, a, 'r', label='Loss Finetunning')
epoch = range(1, len(b) + 1, 1)
plt.plot ( epoch, b, 'g', label='Loss Transfer Learning')
epoch = range(1, len(c) + 1, 1)
plt.plot ( epoch, c, 'b', label='Loss From Scratch')
#plt.plot ( epochs, data['loss_v'],  'b', label='Loss avg')
plt.title ('Losses FCOS')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend()
plt.figure()
plt.show()