import json

filename = '/home/bringascastle/Documentos/repos/SSD/results/losses.json'

with open(filename, "r") as file:
    datos = json.load(file)


def parserFloat(array):
    return [float(v) for v in array] 

print("NMS: ", len(datos['NMS']))
print("El promedio NMS: {}".format(sum(parserFloat(datos['NMS']))/len(datos['NMS'])))

print("DRAW: ", len(datos['draw']))
print("El promedio draw: {}".format(sum(parserFloat(datos['draw']))/len(datos['draw'])))

print("GENERAL: ", len(datos['general']))
print("El promedio general: {}".format(sum(parserFloat(datos['general']))/len(datos['general'])))

print("RED: ", len(datos['red']))
print("El promedio red: {}".format(sum(parserFloat(datos['red']))/len(datos['red'])))