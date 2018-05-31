
![El Detectozord](https://github.com/alleboudy/pointnet/blob/master/doc/eldetectozord.png?raw=true "El detectoZord")

## This repository is a toolbox for using pcl and tensorflow in computer vision(Apologies for the experimental code)
### "Segmentation" is a pipeline that utilizes pcl to grab 3d pointcloud from a depth sensor, apply SAC segmentation to remove large flat surfaces, exctract candidate instances from the scene via pcl euclidean cluster extraction, classify them using pointnet into predefined set of objects then estimate their poses through pcl SampleConsensusPrerejective 
### running segmentation requires first clone the pointnet forked repository "please read down for more info" into segmentation/data/pointnet and  running the flask classification server onlineClassify.py 
## prerequisites used:
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

### Running the Segmentation pipeline
```

1- install tensorflow v 1.0  was used [pip install tesorflow==1.1], later versions might have a problem restoring the checkpoints

2- pip install requests  [needed for calling the online classifier]

3- install opencv, PCL and its dependencies [for windows useres, check out: http://unanancyowen.com/en/pcl181]

4- run onlineClassify.py [the classification flask app] [feel free to change the pipelineCode in the script to change the model used <currently 2 is not available!>]

5- build and run segmentation, to switch it to a realtime set the boolean flag in the main.cpp live=true;

[yup, I need to clean that and get the flags out as parameters, sorry xD]
```


## TODO: refine the pose estimation step and report only the yml of the highest convergence score.
Also, cleaning up and moving flags outside of the code ex: in main.cpp live=true will grab RGB and depth images from a connected OpenNI sensor instead of loading static scenes 

## PointNet work is being done here ->
https://github.com/alleboudy/pointnet Classification, OneVsAllClassification, and pose regresssion

## In /utils you can fibd methods to create point clouds .ply files from given RGB D images, it does the translation to origin and sets the clouds in unit bounding box which is a mandatory preprocessing step for pointnet training.

## more details about the exercises will be added later 
### Challenge 1 notes - UPDATED:
Unfortunately, we haven't finished the classifier yet

please disable the viewers if necessary for a faster running, sorry didn't make it in time before the deadline

please don't mind the correspondeces viewer, it is broken

We didn't find fine parameters for the can, will hopefully do for the fuurther steps

Thank you.



### Challenge1 notes:


#### Strategy:

Our pipeline under challenge_I is used to detect models in given scenes

For each model, a set of parameters are to be tuned

Given a model to detect, we use PointNet to first classify and recognize the model to load the correct set of parameters for the detection pipeline


#### TODO(in progress):

Train the PointNet model on the data provided under train

1-First in the main directory, util is a tool to turn the depth and rgb images we have per model into pointclouds in .ply files to be used to train PointNet

2-Using the script under pointnet/utils/ challenge_prep.py one can generate .h5 of the .ply files we have to be used for training and testing 

3-training pointnet on our data and attaching a script to classify clouds at the top of the detection pipeline to be called to load the right parameters

5- still need to find good parameters for house, bond, pot and shoe :S


#### TODO(next):

We are matching 3d models to 2.5d scenes, which is not practically good, we need to instead take parts of the models for the matching, 

one proposed strategy is: slicing the model on the model one time per axis and taking the resulting half for the matching [means we will match 6 different halves of the model]

add normals and RGB data to pointnet

### Exercise7 notes:

After multiple tries we settled on using global hypothesis verification instead of the greedy one 
the implementation of greedy is left commented in both problems for validation
We still need to better tune the parameters of the pipeline, however. 

### Exercise6 notes:
issues with the parameters are fixed for Ex6.2 and it perfectly aligens, however, ex6.3 aren't perfectly aligned unfortunately, would appreciate any pointers or guidelines, thanks.


### Exercise5 notes:
data/outcomes includes sample outcomes

### Exercise4 notes:
Visualization.ipynb is for problem number 4 Feature Visualization in the exercise

under ssd/ is demo.py for problem number 5

### Exercise3 notes:
the training notebook now is up to date, also the prediction notebook,
however training.py is used for training and modifiedutils, is using python 2 , apologies for this

### Exercise2 notes:
Please note that:
THRESHOLD variable is defined and used to skip further processing if the number
of inlier matches is less than it, we found that using a radically different tem
plate from the image in the scene results in very little matching keypoints and thus poor homography


