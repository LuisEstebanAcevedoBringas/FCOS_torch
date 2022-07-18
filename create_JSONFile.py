import xml.etree.ElementTree as ET
from glob import glob
import json
import os

labels = {-1: 'other', 0: 'cloth', -1: 'other', 1: 'none', 2: 'respirator', 3: 'surgical', 4: 'valve'}

conjunto = 'TEST'  # Para saber que tipo de conjunto se va a pasar json
roots = '/Users/agustincastillo/Downloads/JPEGImages04res/'  # Raiz de la carpeta
carpeta_imagenes = 'JPEGImages04res'  # Carpeta de las imagenes

def getPathImgXML(root, annotations='annotations', images='images'):
    '''
    Params:
    root (ruta raiz donde estan las carpetas de imagenes y anotaciones) 
    annotations (nombre de la carpeta de las anotaciones)
    images (nombre de la carpeta de las imagenes)
    Obtendremos todas las ubicaciones en orden alfabetico
    '''
    path = root + annotations + '/*.xml'
    annot = glob(path)
    annot.sort()

    path = root + images + '/*.jpg'
    imgs = glob(path)
    imgs.sort()

    if len(annot) == 0 or len(imgs) == 0:
        print('No hay imagenes')
        return

    print('Hay {} imagenes y anotaciones'.format(len(imgs)))
    # Aqui vamos a descartar las imagenes inapropiadas y los errores de deteccion
    inapropiate = 0
    errors = 0
    for i, xml in enumerate(annot):
        tree = ET.parse(xml)
        root = tree.getroot()

        if int(root.find('faces').text) == -1:
            annot.pop(i)
            imgs.pop(i)
            inapropiate += 1

        if int(root.find('faces').text) == 0:
            annot.pop(i)
            imgs.pop(i)
            errors += 1

    print("Se encontraron {} imagenes inapropiadas y {} errores de deteccion". format(inapropiate, errors))
    return annot, imgs

def generateJSONFile():
    '''
    Despues generaremos primero el json de las imagenes
    '''
    annot, imgs = getPathImgXML(roots, annotations='Annotations', images=carpeta_imagenes)

    full = roots + 'JSONFiles'
    os.makedirs(full, exist_ok=True)

    file_name = conjunto + '_images.json'

    with open(os.path.join(full, file_name), 'w') as file:
        json.dump(imgs, file)

    print('Hay {} imagenes'.format(len(imgs)))

    file_name = conjunto + '_objects.json'

    with open(os.path.join(full, file_name), 'w') as file:
        json.dump(getObjectsForXML(annot), file)

def getObjectsForXML(annot):

    out = []
    objects = 0

    map = {
        'surgical': 0,
        'valve': 0,
        'cloth': 0,
        'respirator': 0,
        'none': 0,
        'other': 0
    }

    for xml in annot:
        tree = ET.parse(xml)
        root = tree.getroot()
        dics, acum = toDictionary(root)
        objects += len(acum)

        for i in acum:
            map[labels[i]] += 1

        out.append(dics)

    print('Hay {} objetos en el dataset.'.format(objects))
    print('Objetos por clase\n')
    for i, j in zip(map.values(), map.keys()):
        print('{} : {}'.format(j, i))
    return out

def toDictionary(root):
    boxes = []
    labels = []

    for object in root.iter('object'):
        label = object.find('label').text
        bbox = object.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(label))

    return {'boxes': boxes, 'labels': labels}, labels


generateJSONFile()