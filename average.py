import json

filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Results_TransferLearning.json' #TransferLearning
#filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Results_FineTuning.json' #FineTuning
#filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Results_FromScratch.json' #FromScratch

with open(filename, "r") as file:
    datos = json.load(file)

def parserFloat(array):
    return [float(v) for v in array] 

print("NMS: ", len(datos['NMS']))
print("El tiempo promedio NMS es: {}".format(sum(parserFloat(datos['NMS']))/len(datos['NMS'])))

print("DRAW: ", len(datos['draw']))
print("El tiempo promedio del draw es: {}".format(sum(parserFloat(datos['draw']))/len(datos['draw'])))

print("GENERAL: ", len(datos['general']))
print("El tiempo promedio general es: {}".format(sum(parserFloat(datos['general']))/len(datos['general'])))

print("RED: ", len(datos['red']))
print("El tiempo promedio de la red es: {}".format(sum(parserFloat(datos['red']))/len(datos['red'])))