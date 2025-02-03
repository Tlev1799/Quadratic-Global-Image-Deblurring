# Quadratic-Global-Image-Deblurring
Learning a global transformation for deblurring images.

Based on the paper:

Supervised and Unsupervised Learning of Parameterized Color Enhancement https://arxiv.org/abs/2001.05843

Yoav Chai, Raja Giryes, Lior Wolf [Supervised and Unsupervised Learning of Parameterized Color Enhancement]([https://arxiv.org/abs/2001.05843])

## Background
The problem of deblurring images is not new; There are several architectures that have proved efficient in this regard, some of them are UNET based (for example UAE, which is UNET with attention), some diffusion models, and there are many more.
However, most of these models use a lot of resources and computation power, relying on detecting specifics in the image (like edges, items, people) and on local transformations. In this project we used a different approach.
Since blurring is usually caused by the same noise in the entire picture (phone shaking, movement of objects in the frame), perhaps with enough parameters it's possible to construct a global transformation with a spatial dependence to do the job.


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.12.8 (Anaconda)`|
|`torch`|  `2.5.1`|
|`torchvision`|  `0.20.1`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`deblurring.py`| Training the model|
|`utils.py`| utility functions|
|`test_model.py`| Test trained model on test set|
|`Heaper-network Deblurring presentaion.pptx`| PowerPoint Presentation|


## Usage

### Training
First, train the model with `deblurring.py` file. Make sure you downloaded the Gopro dataset from:
[Deblurring on GoPro]([https://paperswithcode.com/sota/deblurring-on-gopro]).

Update the paths to the dataset and path to save the trained models, inside the `deblurring.py` file.

Then run `python deblurring.py`, no parameters requried.


### Testing
Update the path to the model you saved and to the dataset, in file `test_model.py`.
Make sure the path to save the results exists.
Run `python test_model.py`, again no parameters required.


## References
* Yoav Chai, Raja Giryes, Lior Wolf [Supervised and Unsupervised Learning of Parameterized Color Enhancement]([https://arxiv.org/abs/2001.05843])
* [Deblurring on GoPro]([https://paperswithcode.com/sota/deblurring-on-gopro])



