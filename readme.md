# FCOS - torch

## Instrucciones

- Para generar los archivos json del MM dataset se de utilizar el script **create_JSONFIle.py** y se debe de cambiar:
   - linea 8: el conjunto se debe de actualizar segun la carpeta (Valid, Test, Train)
   - linea 9: se debe de pasar la ruta de la carpeta raiz
   - linea 10: se debe de poner el nombre de la carpeta con las imagenes
   - La carpeta con los json se genera en la carpeta donde estan las imagenes y las anotaciones

- Para entrenar se debe de ejecutar el script **train.py** y se debe de cambiar:
   - linea 50: se debe de poner la ruta de la carpeta donde estan los json 

- Para guardar la loss de entrenamiento se debe de actualizar al linea 7 del script **engine.py** con la ruta del archivo **Losses_FCOS_MM_Dataset.json**