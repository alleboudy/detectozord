from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
from random import shuffle
import numpy as np
import h5py

from os import listdir
from os.path import isfile, join


mainplyDir='D:\\plarr\\train2'
plyfiles2load=[f for f in listdir(mainplyDir) if isfile(join(mainplyDir, f))]
#['bird-.ply','bond-.ply','can-.ply','cracker-.ply','shoe-.ply','teapot-.ply']
outputh5FilePath='C:\\Users\\ahmad\\Desktop\\pointnetchallengedata\\RGBDtrain.h5'



# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, 
        data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()
# Load PLY file
def load_ply_data(filename):
    try:
        
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        pcnxyz_array=[]
        sampled_pcxyz_array=[]
        sampled_pcnxyz_array=[]
        for w in pc:
            x=w[0]
            y=w[1]
            z=w[2]
            pcxyz_array.append([x, y, z])
            #pcnxyz_array.append([_nx,_ny,_nz])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
           # sampled_pcnxyz_array.append(pcnxyz_array[i])

        return np.asarray(sampled_pcxyz_array),np.zeros_like(sampled_pcxyz_array)
    except :
        print('err loading file')



labelsMap = dict({"bird":0,"bond":1,"can":2,"cracker":3,"house":4,"shoe":5,"teapot":6})

allpoints=[]
allnormals=[]
alllabels=[]
counter=0
for plyFile in plyfiles2load:
    print(plyFile)
    counter+=1
    print("file number: ",counter)
    try:
        plyxyz,plynxyz = load_ply_data(join(mainplyDir,plyFile))
        allpoints.append(plyxyz)
        allnormals.append(plynxyz)
        alllabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    except:
        print('err loading file')
        continue


indices=list(range(len(allpoints)))
shuffle(indices)

allpoints_shuffle = [allpoints[i] for i in indices] 
allnormals_shuffle = [allnormals[i] for i in indices] 
alllabels_shuffle = [alllabels[i] for i in indices] 



save_h5_data_label_normal(outputh5FilePath,np.asarray(allpoints_shuffle),np.asarray(alllabels_shuffle),np.asarray(allnormals_shuffle))