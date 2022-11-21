from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import cv2


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        OBJECT_SET_CLASSES = [
    'cone',           #0
    'cross_lines',    #1
    'curved_surface', #2
    'cylinder',       #3
    'cylinder_shell', #4
    'cylinder_side',  #5
    'dot_in',         #6
    'dots',           #7
    'flat_slab',      #8
    'hexagon',        #9
    'line',           #10
    'moon',           #11
    'pacman',         #12
    'parallel_lines', #13
    'prism',          #14
    'random',         #15
    'sphere2',        #16
    'sphere',         #17
    'torus',          #18
    'triangle',       #19
    'wave1'           #20
    ]

        # np.random.seed(1)
        
        for i in range(21):
            test_sample = np.random.randint(99,size=20)

            for j in range(-1,2):
                for k in range(-1,2):
                    
                    for d in range(1,11):
                        label = np.zeros(21)

                        label[i] = 1
                        num = (j+1)*33+(k+1)*11+d+1

                        filename = OBJECT_SET_CLASSES[i] +"__%d__%d_%d_%d.png"%(num,j,k,d)
                        self.test_dataset.append([filename, label])
                        # if d!=1:
                        #     self.train_dataset.append([filename, label])

                        if num-1 not in test_sample:
                            self.train_dataset.append([filename, label])
                        #     self.test_dataset.append([filename, label])
                        # else:
                        #     self.train_dataset.append([filename, label])

 


        print('Finished preprocessing the dataset...')
        print('train sample number: %d.'%len(self.train_dataset))
        print('test sample number: %d.'%len(self.test_dataset))

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, crop_size=480, image_size=128, 
               batch_size=32,  mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []


    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))

    transform = T.Compose(transform)

    dataset = CelebA(image_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader