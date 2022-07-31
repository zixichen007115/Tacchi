import os
from os import path 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--particle", default="5", choices=["5","20","100"])
args = parser.parse_args()

file  = os.listdir("obj_"+args.particle)
for f in file:

	print(f[:-4])

	for i in [-1,0,1]:
		for j in [-1,0,1]:
			cmd = "python gel_press.py --object " + f[:-4] + " --particle " + args.particle + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)
			cmd = "python particle_to_depth.py --object " + f[:-4] + " --particle " + args.particle + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)
