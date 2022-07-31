import os
from os import path 

file  = os.listdir("obj_5")
for f in file:

	print(f[:-4])

	for i in [-1,0,1]:
		for j in [-1,0,1]:
			cmd = "python gel_press.py --object " + f[:-4] + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)
			cmd = "python particle_to_depth.py --object " + f[:-4] + " --x " + str(i) + " --y " + str(j)
			print(cmd)
			os.system(cmd)