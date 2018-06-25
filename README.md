# Demo

https://youtu.be/ScUGUCLZF0k


![El Detectozord](https://github.com/alleboudy/pointnet/blob/master/doc/eldetectozord.png?raw=true "El detectoZord")

## This repository is a sandbox for using pcl and tensorflow in computer vision (Apologies for the experimental code)
#### "Segmentation" is a pipeline that utilizes pcl to grab 3d pointclouds from a depth sensor, apply SAC segmentation to remove large flat surfaces, exctract candidate instances of predefined 3D models in segmentation/data/models from the scene via pcl euclidean cluster extraction, classify them using pointnet into predefined set of objects then estimate their poses through pcl SampleConsensusPrerejective 

## Packages used:
```
python 3.5 x64

https://www.python.org/downloads/release/python-350/
```
```
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.0-cp35-cp35m-win_amd64.whl
```
```
pip install scipy
```
```
pip install image
```
```
pip install matplotlib
```
```
pip install flask
```
we are also now using Microsot's cpprestsdk to send post requests to the classification server instead of the python requestclassify.py

https://github.com/Microsoft/cpprestsdk/wiki/Getting-Started-Tutorial


### Running the Segmentation pipeline
```

1- install tensorflow v 1.0  was used [pip install tesorflow==1.1], later versions might have a problem restoring the checkpoints

2- pip install requests  [needed for calling the online classifier]

3- install opencv, PCL and its dependencies [for windows useres, check out: http://unanancyowen.com/en/pcl181]

4- run onlineClassify.py [the classification flask app] [feel free to change the pipelineCode in the script to change the model used <currently 2 is not available!>]

5- build and run segmentation, to switch it to a realtime set the boolean flag in the main.cpp live=true;

[yup, I need to clean that and get the flags out as parameters, sorry xD]
```


### TODO: refine the pose estimation step and report only the yml of the highest convergence score.
Also, cleaning up and moving flags outside of the code ex: in main.cpp live=true will grab RGB and depth images from a connected OpenNI sensor instead of loading static scenes 

## PointNet work is being done here ->
https://github.com/alleboudy/pointnet Classification, OneVsAllClassification, and pose regresssion

### In /utils...
One can fibd methods to create point clouds .ply files from given RGB D images, it does the translation to origin and sets the clouds in unit bounding box which is a mandatory preprocessing step for pointnet training.

#### Challenge 1 notes - UPDATED:

please disable the viewers if necessary for a faster running, sorry didn't make it in time before the deadline

please don't mind the correspondeces viewer, it is broken



#### A pure PCL Detection pipeline:

Our pipeline under challenge_I is used to detect models in given scenes

For each model, a set of parameters are to be tuned

Given a model to detect, we use PointNet to first classify and recognize the model to load the correct set of parameters for the detection pipeline


#### Tips for building training sets:

1-In the main directory, util is a tool to turn the depth and rgb images we have per model into pointclouds in .ply files to be used to train PointNet

2-Using the script under pointnet/utils/ challenge_prep.py one can generate .h5 of the .ply files we have to be used for training and testing 



### Exercise7 notes:

Feature matching in 3D and pose estimation

After multiple tries we settled on using global hypothesis verification instead of the greedy one 
the implementation of greedy is left commented in both problems for validation
We still need to better tune the parameters of the pipeline, however. 

### Exercise6 notes:
Point clouds registeration examples
issues with the parameters are fixed for Ex6.2 and it perfectly aligens.


### Exercise5 notes:

Acquiring pointclouds through a kinict sensor using OpenNI2

data/outcomes includes sample outcomes

### Exercise4 notes:
Another dive into tensorflow, focusing on object detection in 2D images
Visualization.ipynb uses a pretrained model to detect objects in images by sliding accorss the images with a dark window and computing the cross entropy between the acquired logits from the original image and the masked image with the window then coloring the original image depending on the entropy value
under ssd/ is demo.py for object detection using SSD

### Exercise3 notes:
A quick dive into tensorflow
the training notebook now is up to date, also the prediction notebook,
however training.py is used for training and modifiedutils, is using python 2 

### Exercise2 notes:
2D flat objects feature matching between a template and an image, detecting features, representing them,matching and computing homography to align the template on the corresponding matched object in the image 

Please note that:
THRESHOLD variable is defined and used to skip further processing if the number
of inlier matches is less than it, we found that using a radically different tem
plate from the image in the scene results in very little matching keypoints and thus poor homography

### Exercise1 notes:
Simple image reading and filters using OpenCV

