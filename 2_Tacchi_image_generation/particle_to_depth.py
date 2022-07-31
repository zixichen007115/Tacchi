import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from depth_generation import generate
import cv2
import os

if not os.path.exists("unaligned/sim"):
    os.makedirs("unaligned/sim")
if not os.path.exists("unaligned/depth"):
    os.makedirs("unaligned/depth")

parser = argparse.ArgumentParser()
parser.add_argument("--object", default="dot_in")
parser.add_argument("--particle", default="5", choices=["5","20","100"])
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--y", type=int, default=0)
args = parser.parse_args()

obj_name = 'surface/' + args.particle + '_' +args.object+'.npz'

data = np.load(obj_name)

p_xpos = data['p_xpos_list']
p_ypos = data['p_ypos_list']
p_zpos = data['p_zpos_list']
print(np.shape(p_zpos))

num_particle = 101
img_length = 640
img_width = 480

p_depth = np.zeros((num_particle,num_particle))
i_depth = np.zeros((img_width,img_length))

xi = np.linspace(10, 30, num=img_length)
yi = np.linspace(12.5, 27.5, num=img_width)

z_ref = 0.03
k=0

for i in range(150):

    xp = p_xpos[i][:10201]
    yp = p_ypos[i][:10201]
    p_depth = (p_zpos[i][:10201]-12)/1000+0.03

    if np.abs(np.min(p_depth)-z_ref)<1e-4 and z_ref>=0.029:

        i_depth = interpolate.griddata((yp, xp), p_depth, (xi[None,:],yi[:,None]), method='cubic', fill_value=0.03)

        i_depth = i_depth.astype(np.float32)
        pos = "__"+str((args.y+1)*33+(args.x+1)*11+k+1)+"__%d_%d_%d"%(args.y,args.x,k)

        npy_name = "unaligned/depth/" +args.object+pos+".npy"
        np.save(npy_name,i_depth)
        
        img = generate(i_depth)

        img_name = "unaligned/sim/" +args.object+pos+".png"


        cv2.imwrite(img_name, img)

        print(k)
        z_ref -=0.0001
        k +=1
