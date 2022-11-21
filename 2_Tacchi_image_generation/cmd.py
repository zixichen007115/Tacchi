import os
from os import path 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--particle", default="1", choices=["1","10","100"])
args = parser.parse_args()

file  = os.listdir("obj_"+args.particle)
for f in file:

	print(f[:-4])

	for i in [-1,0,1]:
		for j in [-1,0,1]:
			cmd = "python3 gel_press.py --object " + f[:-4] + " --particle " + args.particle + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)
			cmd = "python3 particle_to_depth.py --object " + f[:-4] + " --particle " + args.particle + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)
