# FCOS - torch

- Utilizamos el dataset de Pascal Visual Object Classes (VOC) de los años 2007 y 2012.

   - [2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) 
   - [2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
   - [2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

## Instrucciones
### Para cambiar el metodo de entrenamiento se debe cambiar lo siguiente en el archivo ./index.py:
1) Para hacer TransferLearning se debe usar la funcion "FCOS_TransferLearning()" de la linea 57.
2) Para hacer FineTuning se debe usar la funcion "FCOS_FineTuning()" de la linea 58.
3) Para hacer FromScratch se debe usar la funcion "FCOS_FromScratch()" de la linea 59.

- **./index.py**
  - linea 126
    - Si al main() se le pasa el path de un checkpoint se retomará el entrenamiento desde el checkpoint, si no se le pasa nada, empezará desde la época 0

- **./preference/detect/engine.py**
   - linea 13
     - Se debe de actualizar la ruta donde esta el archivo **.json** que guardara los valores de las perdidas
   - linea 50
     - Se debe de poner el nombre del diccionario donde se van a guardar los valores de las perdidas (Este diccionario debe de estar dentro del mismo archivo que mandamos a llamar en la linea 11)

- **./evaluate/eval.py**
   - linea 11
   - linea 16 
     - Se debe de actualizar el path del checkpoint que queremos evaluar.

- **./utils/draw.py**
   - linea 33 y 54
     - Se debe de actualizar el path de donde esta el **.json** 
   - linea 65
     - Se debe de actualizar el path del **checkpoint** que queremos utilizar para dibujar las bounding boxes.
   - linea 78
     - Se debe de actualizar el path de las **imagenes** que queremos utilizar para dibujar las bounding boxes.

- **./average.py**
   - linea 3, 4 o 5
     - Se debe de actualizar el path de los **json** donde se almacenaron los losses del metodo de entrenamiento (TF, FT, FS)

- **./graficar.py**
   - linea 8, 9 y 10
     - Se debe de actualizar de path de los **json** donde se almacenaron los losses del metodo de entrenamiento (TF, FT, FS) para poder graficarlos.