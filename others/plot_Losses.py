import matplotlib.pyplot as plt
import json

epoch = 30

filename = '/home/fp/Escritorio/LuisBringas/FCOS/results_G/Losses_FineTuning_G.json' #FineTuning

with open(filename, "r") as file:
    finetunning = json.load(file)
    # 2. Update json object
#with open(filename2, "r") as file:
#    transfer = json.load(file)
    # 2. Update json object
#with open(filename3, "r") as file:
#    fromscratch = json.load(file)
    # 2. Update json object

def getArray(data):
    a = []
    for i in range(0, len(data), 2069):
        a.append(float(data[i]))
    return a

#epoch = range(1, epoch , 1)

#a = getArray(finetunning['Losses_FineTuning'])
a = getArray(finetunning['Losses_FineTuning_G'])
#a, b, c = getArray(finetunning['Losses_FineTuning']), getArray(transfer['Losses_TransferLearning']), getArray(fromscratch['Losses_FromScratch'])

epoch = range(1, len(a) + 1, 1)
plt.plot ( epoch, a, 'r', label='Loss Losses_FineTuning')
#epoch = range(1, len(b) + 1, 1)
#plt.plot ( epoch, b, 'g', label='Loss Transfer Learning')
#epoch = range(1, len(c) + 1, 1)
#plt.plot ( epoch, c, 'b', label='Loss From Scratch')
#plt.plot ( epochs, data['loss_v'],  'b', label='Loss avg')
#plt.title ('Losses FCOS')
plt.title ('Losses FCOS with new dataset')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend()
plt.figure()
plt.show()