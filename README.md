# Challenge1 notes:

## Strategy:

Our pipeline under challenge_I is used to detect models in given scenes

For each model, a set of parameters are to be tuned

Given a model to detect, we use PointNet to first classify and recognize the model to load the correct set of parameters for the detection pipeline


## TODO(in progress):

Train the PointNet model on the data provided under train

1-First in the main directory, util is a tool to turn the depth and rgb images we have per model into pointclouds in .ply files to be used to train PointNet

2-Using the script under pointnet/utils/ challenge_prep.py one can generate .h5 of the .ply files we have to be used for training and testing 

3-training pointnet on our data and attaching a script to classify clouds at the top of the detection pipeline to be called to load the right parameters

5- still need to find good parameters for house, bond, pot and shoe :S

## TODO(next):

We are matching 3d models to 2.5d scenes, which is not practically good, we need to instead take parts of the models for the matching, 

one proposed strategy is: slicing the model on the model one time per axis and taking the resulting half for the matching [means we will match 6 different halves of the model]




# Exercise7 notes:
After multiple tries we settled on using global hypothesis verification instead of the greedy one 
the implementation of greedy is left commented in both problems for validation
We still need to better tune the parameters of the pipeline, however. 

# Exercise6 notes:
issues with the parameters are fixed for Ex6.2 and it perfectly aligens, however, ex6.3 aren't perfectly aligned unfortunately, would appreciate any pointers or guidelines, thanks.


# Exercise5 notes:
data/outcomes includes sample outcomes

# Exercise4 notes:
Visualization.ipynb is for problem number 4 Feature Visualization in the exercise

under ssd/ is demo.py for problem number 5

# Exercise3 notes:
the training notebook now is up to date, also the prediction notebook,
however training.py is used for training and modifiedutils, is using python 2 , apologies for this

# Exercise2 notes:
Please note that:
THRESHOLD variable is defined and used to skip further processing if the number
of inlier matches is less than it, we found that using a radically different tem
plate from the image in the scene results in very little matching keypoints and thus poor homography


