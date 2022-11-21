import open3d as o3d
import numpy as np
import os
from os import path 


url_pcd = "pcd"

def pcd2npy(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

file  = os.listdir(url_pcd)

for f in file:
    part_npy_file = np.empty([0,3])
    print(f[:-4])

    read_url = path.join (url_pcd , f)
    npy_file = pcd2npy(read_url)

    n_particle = int(np.shape(npy_file)[0]/10000)

    url_npy = "obj_" + str(n_particle)
    if not os.path.exists(url_npy):
        os.makedirs(url_npy)
    

    for i in range(np.shape(npy_file)[0]):
        tmp = np.reshape(npy_file[i,:],(1,3))
        part_npy_file = np.concatenate((part_npy_file,tmp))
        if i%5000==0:
            print(i)
        if npy_file[i,2]>18:
            tmp = np.reshape(npy_file[i,:],(1,3))
            part_npy_file = np.concatenate((part_npy_file,tmp))


    print(np.shape(part_npy_file))
    print("max: %f, %f, %f. min: %f, %f, %f."%(np.max(part_npy_file[:,0]),np.max(part_npy_file[:,1]),
        np.max(part_npy_file[:,2]),np.min(part_npy_file[:,0]),np.min(part_npy_file[:,1]),
        np.min(part_npy_file[:,2])))

    write_url = path.join (url_npy , f[:-3])
    write_url = write_url + "npy"
    np.save(write_url,part_npy_file.astype(np.float32))

