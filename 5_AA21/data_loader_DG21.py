import numpy as np
import os
from image_generation_DG21 import generate
import cv2

file = os.listdir('depth')
image_smoooth = np.zeros([480,640])

for f in file:
    data = np.load('depth/'+f)


    RGB_image_smoooth = generate(data)
    cv2.imwrite('image_DG21/'+f[:-3]+'png',RGB_image_smoooth)
    # cv2.imshow('test',RGB_image_smoooth)
    # cv2.waitKey(5)

            

    print(f[:-4])
    # print(np.min(image_smoooth))




    # sys.exit()

