import os
import sys
import numpy as np
import h5py
from random import shuffle
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#    zipfile = os.path.basename(www)
#    os.system('wget %s; unzip %s' % (www, zipfile))
#    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_ply_data(filename):

        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pcxyz_array=[]
        pcnxyz_array=[]
        sampled_pcxyz_array=[]
        sampled_pcnxyz_array=[]
        for items in pc:
            x=items[0]
            y=items[1]
            z=items[2]
            pcxyz_array.append([x, y, z])
        indices = list(range(len(pcxyz_array)))
        indicessampled= np.random.choice(indices, size=2048)
        for i in indicessampled:
            sampled_pcxyz_array.append(pcxyz_array[i])
        #normalizing and zero center:
        sampled_pcxyz_array = np.asarray(sampled_pcxyz_array)
        #print(sampled_pcxyz_array.shape)
        minx = min(sampled_pcxyz_array[:,0])
        miny = min(sampled_pcxyz_array[:,1])
        minz = min(sampled_pcxyz_array[:,2])
        maxx = max(sampled_pcxyz_array[:,0])
        maxy = max(sampled_pcxyz_array[:,1])
        maxz = max(sampled_pcxyz_array[:,2])
        scale = min((1 / (maxx - minx)), min(1 / (maxy - miny),1/ (maxz-minz)))
        sampled_pcxyz_array[:,0] = (sampled_pcxyz_array[:,0] - 0.5*(minx + maxx))*scale + 0.5
        sampled_pcxyz_array[:,1] = (sampled_pcxyz_array[:,1] - 0.5*(miny + maxy))*scale + 0.5
        sampled_pcxyz_array[:,2] = (sampled_pcxyz_array[:,2] - 0.5*(minz + maxz))*scale + 0.5
        sampled_pcxyz_array[:,0] -= np.average(sampled_pcxyz_array[:,0])
        sampled_pcxyz_array[:,1] -= np.average(sampled_pcxyz_array[:,1])
        sampled_pcxyz_array[:,2] -= np.average(sampled_pcxyz_array[:,2])
        return sampled_pcxyz_array


