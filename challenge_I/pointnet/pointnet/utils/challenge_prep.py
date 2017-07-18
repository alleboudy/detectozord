from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
from random import shuffle
import numpy as np
import h5py

from os import listdir
from os.path import isfile, join
NUMBERofSamplesPerModel = 5000#how many sample clouds of size 2048 to draw per model
SizeOfTestSplit= 1000#how many of the drawn samples is for test
SizeOfValSplit= 1000#how many for validation
mainplyDir='models/'
plyfiles2load=[f for f in listdir(mainplyDir) if isfile(join(mainplyDir, f))]
#['bird-.ply','bond-.ply','can-.ply','cracker-.ply','shoe-.ply','teapot-.ply']
outputh5FilePath='gen'




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
        for x,y,z,_nx,_ny,_nz,_r,_g,_b,_a in pc:
            pcxyz_array.append([x, y, z])
            pcnxyz_array.append([_nx,_ny,_nz])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
            sampled_pcnxyz_array.append(pcnxyz_array[i])

        return np.asarray(sampled_pcxyz_array),np.asarray(sampled_pcnxyz_array)
    except :
        pass




def load_ply_data_manySamples(filename,numberOfSamples):
    #try:
        allSamples_xyz_arrays=[]
        allSamples_normals_arrays=[]
        train_xyz_arrays=[]
        train_normals_arrays=[]
        val_xyz_arrays=[]
        val_normals_arrays=[]
        test_xyz_arrays=[]
        test_normals_arrays=[]
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        pcnxyz_array=[]

        for x,y,z,_nx,_ny,_nz,_r,_g,_b,_a in pc:
            pcxyz_array.append([x, y, z])
            pcnxyz_array.append([_nx,_ny,_nz])
        indices = list(range(len(pcxyz_array)))
        for x in range(numberOfSamples):
           # print('sampling ...') 
            indicessampled= np.random.choice(indices, size=2048)
            sampled_pcxyz_array=[]
            sampled_pcnxyz_array=[]
            for i in indicessampled:
                sampled_pcxyz_array.append(pcxyz_array[i])
                sampled_pcnxyz_array.append(pcnxyz_array[i])
            allSamples_xyz_arrays.append(np.asarray(sampled_pcxyz_array))
            allSamples_normals_arrays.append(np.asarray(sampled_pcnxyz_array))
        allindices = list(range(len(allSamples_xyz_arrays)))
        shuffle(allindices)
        valIndices = allindices[:SizeOfValSplit]
        testIndices = allindices[SizeOfValSplit:SizeOfValSplit+SizeOfTestSplit]
        #print(SizeOfValSplit,SizeOfTestSplit,testIndices)
        trainIndices = allindices[SizeOfTestSplit+SizeOfValSplit:]
        train_xyz_arrays = [allSamples_xyz_arrays[i] for i in trainIndices]
        test_xyz_arrays = [allSamples_xyz_arrays[i] for i in testIndices]
        val_xyz_arrays = [allSamples_xyz_arrays[i] for i in valIndices] 
        train_normals_arrays = [allSamples_normals_arrays[i] for i in trainIndices]
        test_normals_arrays = [allSamples_normals_arrays[i] for i in testIndices]
        val_normals_arrays = [allSamples_normals_arrays[i] for i in valIndices] 
        return train_xyz_arrays,train_normals_arrays,test_xyz_arrays,test_normals_arrays,val_xyz_arrays,val_normals_arrays
    #except :
        #print("err")





labelsMap = dict({"bird":0,"bond":1,"can":2,"cracker":3,"house":4,"shoe":5,"teapot":6})

allpoints=[]
allnormals=[]
alllabels=[]
valpoints=[]
valnormals=[]
vallabels=[]
testpoints=[]
testnormals=[]
testlabels=[]
counter=0
for plyFile in plyfiles2load:
    print(plyFile)
    counter+=1
    print("file number: ",counter)
    #try:
    plyxyz,plynxyz,testxyz,testnxyz,valxyz,valnxyz = load_ply_data_manySamples(join(mainplyDir,plyFile),NUMBERofSamplesPerModel)
    allpoints+=plyxyz
    allnormals+=plynxyz
    valpoints+=valxyz
    valnormals+=valnxyz
    testpoints+=testxyz
    testnormals+=testnxyz
    for i in range(len(plynxyz)):
        alllabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    for i in range(len(testxyz)):
        testlabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    for i in range(len(valnxyz)):
        vallabels.append(np.asarray([labelsMap[plyFile.split('-')[0]]]))
    
    #except:
        #print('errhere')
        #continue


indices=list(range(len(allpoints)))
shuffle(indices)

allpoints_shuffle = [allpoints[i] for i in indices] 
allnormals_shuffle = [allnormals[i] for i in indices] 
alllabels_shuffle = [alllabels[i] for i in indices] 


print(np.asarray(allpoints_shuffle).shape)
save_h5_data_label_normal(outputh5FilePath+'/'+str((NUMBERofSamplesPerModel-SizeOfTestSplit-SizeOfValSplit))+'train.h5',np.asarray(allpoints_shuffle),np.asarray(alllabels_shuffle),np.asarray(allnormals_shuffle))


valindices=list(range(len(valpoints)))
shuffle(valindices)

valpoints_shuffle = [valpoints[i] for i in valindices] 
valnormals_shuffle = [valnormals[i] for i in valindices] 
vallabels_shuffle = [vallabels[i] for i in valindices] 


print(np.asarray(valpoints_shuffle).shape)
save_h5_data_label_normal(outputh5FilePath+'/'+str(SizeOfValSplit)+'val.h5',np.asarray(valpoints_shuffle),np.asarray(vallabels_shuffle),np.asarray(valnormals_shuffle))


testindices=list(range(len(testpoints)))
shuffle(testindices)
print("test indices ",len(testindices) )
testpoints_shuffle = [testpoints[i] for i in testindices] 
testnormals_shuffle = [testnormals[i] for i in testindices] 
testlabels_shuffle = [testlabels[i] for i in testindices] 


print(np.asarray(testpoints_shuffle).shape)
save_h5_data_label_normal(outputh5FilePath+'/'+str(SizeOfTestSplit)+'test.h5',np.asarray(testpoints_shuffle),np.asarray(testlabels_shuffle),np.asarray(testnormals_shuffle))
