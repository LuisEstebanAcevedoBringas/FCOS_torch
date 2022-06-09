# FCOS - torch

- We used the Pascal Visual Object Classes (VOC) dataset - 2007 and 2012.

   - [2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) 
   - [2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
   - [2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

## Instructions
### To change the training method you must change the following in the ./index.py file:
1) To do TransferLearning you must use the function "FCOS_TransferLearning()" from line 57.
2) To do FineTuning you must use the function "FCOS_FineTuning()" in line 58.
3) To do FromScratch you must use the function "FCOS_FromScratch()" of line 59.

- **./index.py**
  - line 132
    - If the main() is given the path of a checkpoint, the training will resume from the checkpoint, if nothing is given, it will start from epoch 0.

- **./preference/detect/engine.py**
   - line 13
     - The path to the **.json** file that will store the values of the losses must be updated.
   - line 50
     - We must put the name of the dictionary where the values of the losses are going to be saved (This dictionary must be inside the same .json file that we call in line 11).

- **./evaluate/eval.py**
   - line 18
     - The path of the checkpoint to be evaluated must be updated.
   - line 35
     - The path where the .json files with the content of the dataset are located must be updated.

- **./utils/draw.py**
   - line 33 and 54
     - The path to the **.json** must be updated. 
   - line 65
     - We must update the path of the **checkpoint** that we want to use to draw the bounding boxes.
   - line 78
     - We must update the path of the **images** that we want to use to draw the bounding boxes.

- **./average.py**
   - line 3, 4 or 5
     - The path of the **json** where the training method losses (TF, FT, FS) are stored must be updated.

- **./graficar.py**
   - line 8, 9 and 10
     - The path of the **json** where the training method losses (TF, FT, FS) were stored must be updated in order to be able to plot them.