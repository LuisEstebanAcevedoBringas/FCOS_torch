# FCOS - torch

## Instrucciones

- Para generar los archivos json del MM dataset se de utilizar el script **create_JSONFIle.py** y se debe de cambiar:
   - linea 8: el conjunto se debe de actualizar segun la carpeta (Valid, Test, Train)
   - linea 9: se debe de pasar la ruta de la carpeta raiz
   - linea 10: se debe de poner el nombre de la carpeta con las imagenes
   - La carpeta con los json se genera en la carpeta donde estan las imagenes y las anotaciones

- Para entrenar se debe de ejecutar el script **train.py** y se debe de cambiar:
   - linea 11: Se debe de actualizar la ruta de donde se va a guardar el checkpoint que se genera epoca tras epoca.
   - linea 13: Se debe de actualizar la truta de donde se van  a guardar los checkpoints en las epocas 5, 10, 15, 20, 21, 24, 25, 26, 27, 28, 29 y 30.
   - linea 50: se debe de poner la ruta de la carpeta donde estan los json 
   - linea 147: si se quiere utilizar un checkpoint que se guardo previamente (i.e. checkpoint_epoca_5) se debe de pasarle a la funcion main la ruta del checkpoint.

- Para guardar la loss de entrenamiento se debe de actualizar al linea 7 del script **engine.py** con la ruta del archivo **Losses_FCOS_MM_Dataset.json**