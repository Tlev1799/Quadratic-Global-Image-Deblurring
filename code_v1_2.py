# -*- coding: utf-8 -*-

#!pip install nvidia-dali-cuda117

#!/usr/bin/env python
# coding: utf-8



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import torchvision
import random
import os
from PIL import Image
import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import functools
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
# import kornia
import types
import collections
import pandas as pd
from torch.utils import data
from random import shuffle
#from nvidia.dali.plugin.pytorch import DALIGenericIterator
import types
import collections
import numpy as np
from random import shuffle
#from nvidia.dali.pipeline import Pipeline
#import nvidia.dali.ops as ops
#import nvidia.dali.types as types
import time
import timeit
import math
import pdb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#get_ipython().run_line_magic('matplotlib', 'inline')

torch.__version__

# Code taken from kornia
def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    See :class:`~kornia.color.RgbToXyz` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ.

    Returns:
        torch.Tensor: XYZ version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack((x, y, z), -3)

    return out

def xyz_to_lab(image: torch.Tensor) -> torch.Tensor:

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    l: torch.Tensor = (116 * x) - 16
    a: torch.Tensor = 500 * (x - y)
    b: torch.Tensor = 200 * (y - z)

    out: torch.Tensor = torch.stack((l, a, b), -3)

    return out

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v

    s[torch.isnan(s)] = 0.

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h
    return torch.stack([h, s, v], dim=-3)

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to LAB.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))


    return xyz_to_lab(rgb_to_xyz(image))

def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to grayscale.

    See :class:`~kornia.color.RgbToGrayscale` for details.

    Args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


import os


#load data

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a common size
    transforms.ToTensor()         # Convert to a PyTorch tensor
])

transform_no_resize = transforms.Compose([ transforms.ToTensor()])



def load_and_resize_images(folder_path):
    """
    Load images from a folder, resize them to 256x256, and convert them to PyTorch tensors.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: A list of PyTorch tensors representing the resized images.
    """
    image_tensors = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            try:
                # Open the image
                img = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB
                # Apply transformations
                tensor = transform(img)
                image_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return image_tensors

def load_images_without_resizing(folder_path):
    """
    Load images from a folder and convert them to PyTorch tensors without resizing.
    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        list: A list of PyTorch tensors representing the original images.
    """
    image_tensors = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            try:
                # Open the image
                img = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB
                # Apply the transformation (convert to tensor only)
                tensor = transform_no_resize(img)
                image_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return image_tensors

def load_and_resize_to_match(folder_path, reference_images):
    """
    Load images from a folder and resize each one to match the size of the corresponding image in a reference list.
    Args:
        folder_path (str): Path to the folder containing images.
        reference_images (list): A list of PyTorch tensors to match the size of.
    Returns:
        list: A list of PyTorch tensors resized to match the reference images.
    """
    resized_images = []
    for i, file_name in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            try:
                # Open the image
                img = Image.open(file_path).convert('RGB')  # Ensure 3-channel RGB
                # Get the target size from the corresponding reference image
                target_size = reference_images[i].shape[-2:]  # (height, width)
                # Resize the image
                img_resized = img.resize(target_size[::-1], Image.ANTIALIAS)  # Reverse order for (width, height)
                # Convert to tensor
                tensor = transform_no_resize(img_resized)
                resized_images.append(tensor)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return resized_images


#TODO- CHANGE TO YOUR PATH
train_clean_dir = '/ds/users/yaelsf/projects/asr-alpha-edit/michael/datasets/train_clean' # Path to source images
train_blur_dir = '/ds/users/yaelsf/projects/asr-alpha-edit/michael/datasets/train_blur'   # Path to target images

#blur_dir = 'C:\Users\themi\OneDrive - Technion\technion deep learning code\project\DIV2K_train_LR_bicubic_X2'
# clean_dir = '/path/to/your/target/images'


# blur_resized_images = load_and_resize_images(blur_dir)

# clean_images = load_images_without_resizing(clean_dir)

# resized_to_match = load_and_resize_to_match(blur_dir, clean_images)

# # Convert the list of tensors to a batched tensor (optional)
# batched_images = torch.stack(blur_resized_images)  # Shape: [N, C, 256, 256]

class Dataset(data.Dataset):
    def __init__(self, blur_dir, clean_dir):
        self.blur_dir = blur_dir
        self.target_dir = clean_dir
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to a common size
            transforms.ToTensor()
        ])
        self.transform_no_resize = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Get file pairs
        self.dataset = []
        clean_images = [f for f in os.listdir(clean_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
        for clean_file in clean_images:
            blur_path = os.path.join(blur_dir, clean_file.replace('.png', 'x2.png'))
            clean_path = os.path.join(clean_dir, clean_file)
            if os.path.exists(blur_path):
                self.dataset.append((blur_path, clean_path))
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        blur_path, clean_path = self.dataset[index]
        
        # Load images
        blur_img = Image.open(blur_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        resize_blur_tensor = self.transform(blur_img)
        
        clean_tensor = self.transform_no_resize(clean_img)
        target_size = clean_tensor.shape[-2:]  # (height, width)
        # Resize the image
        img_resized = blur_img.resize(target_size[::-1], Image.NEAREST)  # Reverse order for (width, height)
        blur_tensor = self.transform_no_resize(img_resized)
        
        return blur_tensor, resize_blur_tensor, clean_tensor

train_dataset = Dataset(train_blur_dir, train_clean_dir)

# from google.colab import drive
# drive.mount('/content/drive')

#model

class Transformer(nn.Module):
    def __init__(self,Norm=nn.InstanceNorm2d, Act=nn.ReLU(True),in_features=64, in_channels=3, n_batch=5):
        super(Transformer, self).__init__()
#         Act = functools.partial(Act)
        Middle_layer = [nn.Conv2d(in_features*2, in_features*2, 3,2),
                        Norm(in_features*2),
                        Act,
                        nn.Dropout2d(0.5)]
        ml = []
        for i in range(0,3):
            ml+=Middle_layer

        self.Middle_layer = nn.Sequential(*ml)

        self.Input_layer = nn.Sequential(nn.Conv2d(in_channels, in_features, 5,2),
                Act,
                nn.Conv2d(in_features, in_features*2, 3),
                Norm(in_features*2),
                Act
                 )

        self.Average_layer = nn.Sequential(
            nn.Conv2d(in_features*2, in_features*2, 3,2),
            Norm(in_features*2),
            Act,
            nn.AdaptiveAvgPool2d((6, 6)),
            Act,
            nn.Dropout2d(0.5)
        )

        self.Finale_layer = nn.Sequential(
            nn.Conv2d(in_features*2, in_channels, 1),
            Act,
            nn.Dropout2d(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(36,16),
            Act,
            nn.Linear(16,10)
        )
        self.l = nn.Linear(5*n_batch,1*n_batch)

    def forward(self, x,y):
        b1_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b1 = self.classifier(b1_.view(b1_.size(0),3,-1))
        b2_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b2 = self.classifier(b2_.view(b2_.size(0),3,-1))
        b3_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b3 = self.classifier(b3_.view(b3_.size(0),3,-1))
        b4_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b4 = self.classifier(b4_.view(b4_.size(0),3,-1))
        b5_ = self.Finale_layer(self.Average_layer(self.Middle_layer(self.Input_layer(x))))
        b5 = self.classifier(b5_.view(b5_.size(0),3,-1))

        concat = torch.cat([b1,b2,b3,b4,b5],0)
#         print(concat.shape)
        h_theta = self.l(concat.T)
        h_theta = h_theta.permute(0,2,1)
        V_p = self.get_param(y)
        output= torch.einsum("abcde,abf->bfde", (V_p, h_theta))
        return output+y


    def get_param(self,x):
        R = x[:,:1,:,:]
        G = x[:,1:2,:,:]
        B = x[:,2:,:,:]
        C = torch.ones_like(R)
        return torch.stack([R, G, B, torch.pow(R,2), torch.pow(G,2), torch.pow(B,2), R * G, G * B, B * R, C])


class Loss():
    def __init__(self, col_hsv=False, col_lab=True, col_gray=True):
        self.transform = list()
        if col_hsv:
            self.transform.append(rgb_to_hsv)
        if col_lab:
            print("lab")
            self.transform.append(rgb_to_lab)
        if col_gray:
            print("gray")
            self.transform.append(rgb_to_grayscale)

        self.criterion = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
#         self.psnr = kornia.losses.PSNRLoss(2)
#         self.ssim = kornia.losses.SSIM(5, reduction='none')
    def __call__(self,img_input,img_target):
        loss = 0.0
        input = self.transform[0](img_input)
        target = self.transform[0](img_target)
        loss+=20*(self.criterion(input[:,0,:,:],target[:,0,:,:])+self.criterion(input[:,1,:,:],target[:,1,:,:])+self.criterion(input[:,2,:,:],target[:,2,:,:]))

#         loss+=self.criterion(self.transform[1](img_input),self.transform[1](img_target))

        loss+=self.huber(self.transform[1](img_input),self.transform[1](img_target))
        return loss#+self.psnr(img_input,img_target)+self.ssim(img_input,img_target)


epoch = 0
n_epochs = 500
decay_epoch = 30
batchSize = 25
lr = 0.0009


criterion = Loss()


model = Transformer(n_batch=batchSize).cuda()


lambda1 = lambda epoch: 0.55 ** (epoch)
optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.5, 0.999))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


def preimg(img, target):
    resized_images = []
    resized_real = []
    fixed_size = (256, 256)
    
    # Use torchvision's interpolate instead of skimage resize
    resized_images = [F.interpolate(image.unsqueeze(0), size=fixed_size, mode="bilinear", align_corners=False).squeeze(0) 
                      for image in img]

    for source_img, target_img in zip(img, target):
        target_size = target_img.shape[-2:]  # (height, width)
        resized_real_img = F.interpolate(source_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)
        resized_real.append(resized_real_img.squeeze(0))
        
    return resized_images, resized_real

def normalize(img,real,targ):
    img = img.permute(0,3,1,2)
    real = real.permute(0,3,1,2)
    targ = targ.permute(0,3,1,2)
    mean_512 = torch.ones_like(img)*0.5*255
    std_512 = torch.ones_like(img)*0.5*255
    mean_256 = torch.ones_like(real)*0.5*255
    std_256 = torch.ones_like(real)*0.5*255

    img = (img-mean_512)/std_512
    real = (real-mean_256)/std_256
    targ = (targ-mean_512)/std_512

    return img,real,targ


dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

model.train()
for epoch in range(0, n_epochs):
    torch.cuda.empty_cache()
    avg_loss = 0
    for i, batch_data in enumerate(dataloader):
        start = time.time()
        blur_img,resize_blur_img,clean_img = batch_data[0], batch_data[1], batch_data[2]

        blur_img,resize_blur_img,clean_img = normalize(blur_img,resize_blur_img,clean_img)

        optimizer.zero_grad()

        #Inverse identity
        res = model(blur_img,resize_blur_img) # previously for res 3, mask,src
        loss = criterion(res,clean_img)
        loss.sum().backward()

        optimizer.step()


        #############################################
        avg_loss = (avg_loss+loss.sum().item())/(i+1)

        if (i + 1) % 200 == 0:
            stop = time.time()
            print('Time: ', stop - start)
            with open('cpe_64.txt', 'a') as f:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) LR: (%0.0007f) Time: (%0.0007f)" % (epoch, i + 1, 200, avg_loss, optimizer.param_groups[0]['lr'],stop - start),file=f)


        if (i + 1) % 200 == 0:
            pic = (torch.cat([img,res, targ], dim=0).data + 1) / 2.0
            save_dir = "./results"
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, 200), nrow=3)
    if (epoch+1)%1==0:
        torch.save(model.state_dict(), './models/model_64_{}.pth'.format(epoch))
    # Update learning rates
    if (epoch+1)%30==0 and epoch<320:
        lr_scheduler.step()