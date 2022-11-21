import open3d as o3d
import os
from os import path
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--particle", type=int , default=1, choices=[1,10,100])
args = parser.parse_args()

url_ply = "ply"
url_pcd = "pcd"

if not os.path.exists(url_pcd):
	os.makedirs(url_pcd)

file  = os.listdir(url_ply)
for f in file:
	print(f[:-4])
	read_url = path.join (url_ply , f)
	mesh = o3d.io.read_triangle_mesh(read_url) 
	write_url = path.join (url_pcd , f[:-3])
	write_url = write_url + "pcd"
	n_sam = 10000 * args.particle
	cmd = 'pcl_mesh_sampling ' + read_url + ' ' + write_url +" -n_samples " +str(n_sam)

	print(cmd)
	os.system(cmd)
