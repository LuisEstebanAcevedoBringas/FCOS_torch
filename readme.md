# FCOS - torch

- Utilizando el dataset "Pascal VOC"

## Instrucciones

### Para cambiar el metodo de entrenamiento se debe cambiar lo siguiente en el archivo ./index.py:
1) Para hacer TransferLearning se debe usar la funcion "FCOS_TransferLearning()" de la linea 57.
2) Para hacer FineTuning se debe usar la funcion "FCOS_FineTuning()" de la linea 58.
3) Para hacer FromScratch se debe usar la funcion "FCOS_FromScratch()" de la linea 59.

- engine.py
   - linea 16 

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

