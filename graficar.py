import matplotlib.pyplot as plt
from calendar import EPOCH
import json

epoch = 232

filename2 = '/home/bringascastle/Escritorio/resultados_ssd_lite_transfer/losses.json' #TransferLearning
filename1 = '//home/bringascastle/Escritorio/resultados_ssd_lite_finetunning/losses.json' #FineTuning
filename3 = '/home/bringascastle/Documentos/repos/SSD/results/losses_FROM.json' #FromScratch

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

a, b, c = getArray(finetunning['loss_value']), getArray(transfer['loss_value']), getArray(fromscratch['loss_value'])

epoch = range(1, len(a) + 1, 1)
plt.plot ( epoch, a, 'r', label='Loss Finetunning' )
epoch = range(1, len(b) + 1, 1)
plt.plot ( epoch, b, 'b', label='Loss Transfer Learning' )
epoch = range(1, len(c) + 1, 1)
plt.plot ( epoch, c, 'g', label='Loss From Scratch' )
#plt.plot ( epochs, data['loss_v'],  'b', label='Loss avg')
plt.title ('Losses FCOS')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()