import matplotlib.pyplot as plt
import json

filename = 'losses.json'
with open(filename, "r") as file:
    datos = json.load(file)
    

def get_lists_for_plt(array):
    aux = []

    for i in range(59, len(array) + 1, 60):
        aux.append(float(array[i]))

    return aux

l1 = get_lists_for_plt(datos['loss_value'])
l2 = get_lists_for_plt(datos['loss_value_transfer'])
l3 = get_lists_for_plt(datos['loss_value_fine'])

epochs = 232

epochs = range(1, epochs + 1, 1)

plt.plot(epochs, l1, 'r', label='loss from scratch')
plt.plot(epochs, l2, 'b', label='loss transfer learning')
plt.plot(epochs, l3, 'g', label='loss finetuning')

plt.title('Losses SSD')

plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend()
plt.figure()
plt.show()
