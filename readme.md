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

- ./preference/detect/engine.py
   - linea 11
     - Se debe de actualizar la ruta donde esta el archivo **.json** que guardara los valores de las perdidas
   - linea 51
     - Se debe de poner el nombre del diccionario donde se van a guardar los valores de las perdidas (Este diccionario debe de estar dentro del mismo archivo que mandamos a llamar en la linea 11)

- evaluate/detect/eval.py
   - linea 11
   - linea 16 

- ./draw.py
   - linea 33
   - linea 54
   - linea 65
   - linea 78

- ./average.py
   - linea 3

- ./graficar.py
   - linea 8
   - linea 9
   - linea 10

