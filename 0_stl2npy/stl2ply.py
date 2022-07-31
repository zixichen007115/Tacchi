import open3d as o3d
import os
from os import path 

url_stl = "stl"
url_ply = "ply"

if not os.path.exists(url_ply):
	os.makedirs(url_ply)

file  = os.listdir(url_stl)
for f in file:
	print(f[:-4])
	read_url = path.join (url_stl , f)
	mesh = o3d.io.read_triangle_mesh(read_url) 
	write_url = path.join (url_ply , f[:-3])
	write_url = write_url + "ply"
	o3d.io.write_triangle_mesh(write_url,mesh)