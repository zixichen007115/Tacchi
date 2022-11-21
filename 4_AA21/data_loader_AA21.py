import numpy as np
import os
from image_generation_AA21 import generate
import cv2

file = os.listdir('depth')

kernel = np.zeros([13,13])
E=2.7182818285

for i in range(-6,7):
    for j in range(-6,7):
        i_=np.abs(i)
        j_=np.abs(j)

        kernel[(i+6),(j+6)]=201/(200+np.exp(np.sqrt(np.square(i_)+np.square(j_))))

kernel = kernel/np.sum(kernel)


image_smoooth = np.zeros([480,640])



for f in file[::-1]:
    data = np.load('depth/'+f)


    for i in range(480):
        for j in range(640):
            if data[i,j]>0.03:
                data[i,j]=0.03

    image = np.ones([480+12,640+12])*0.03
    image[6:486,6:646] = data

    for i in range(480):
        for j in range(640):
            image_smoooth[i,j]=np.sum(kernel*image[i:i+13,j:j+13])
    
    RGB_image_smoooth = generate(image_smoooth)
    # print(np.dtype(image_smoooth[0,0]))

    np.save('depth_AA21/'+f,np.float32(image_smoooth))
    cv2.imwrite('image_AA21/'+f[:-3]+'png',RGB_image_smoooth)
    # cv2.imshow('test',RGB_image_smoooth)
    # cv2.waitKey(5)

            

    print(f[:-4])
    # print(np.min(image_smoooth))




    # sys.exit()

