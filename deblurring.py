import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision
from PIL import Image

from utils import rgb_to_hsv, rgb_to_lab, rgb_to_grayscale, calculate_ssim_torch


import matplotlib.pyplot as plt
import cv2
import numpy as np

class GoProDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, resize=(256, 256)):
        self.blur_dir = os.path.join(root_dir, split, 'blur')
        self.sharp_dir = os.path.join(root_dir, split, 'sharp')
        self.image_filenames = sorted(os.listdir(self.blur_dir))
        self.transform = transform
        self.resize = resize
        self.resize_transform = transforms.Resize(resize)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.image_filenames[idx])
        sharp_path = os.path.join(self.sharp_dir, self.image_filenames[idx])

        blur_image = Image.open(blur_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        # Resize the blurred image.
        resized_blur_image = self.resize_transform(blur_image)

        if self.transform:
            blur_image = self.transform(blur_image)
            resized_blur_image = self.transform(resized_blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, resized_blur_image, sharp_image

# Define the model and loss functions
class Transformer(nn.Module):
    def __init__(self,Norm=nn.InstanceNorm2d, Act=nn.ReLU(True),in_features=64, in_channels=3, n_batch=7):
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
            nn.AdaptiveAvgPool2d((9, 9)),
            Act,
            nn.Dropout2d(0.5)
        )

        self.Finale_layer = nn.Sequential(
            nn.Conv2d(in_features*2, in_channels, 1),
            Act,
            nn.Dropout2d(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(81,72),
            Act,
            nn.Linear(72,65)
        )
        self.l = nn.Linear(65*5,65)

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

        concat = torch.cat([b1.permute(2,0,1),b2.permute(2,0,1),b3.permute(2,0,1),b4.permute(2,0,1),b5.permute(2,0,1)],0)
        h_theta = self.l(concat.permute(2,1,0))
        h_theta = h_theta.permute(1,0,2)
        B,C,H,W= y.shape

        h1_theta=h_theta[:,:,10:] #saptial H
        h2_theta=h_theta[:,:,:10] #colord H

        #h1_theta=h1_theta.reshape(B*3 , 1, 3, 3) # RGB are 3 chanel
        #y_grouped = y.view(1, B * C, H, W)
        #print(h1_theta.shape)
        #print(y_grouped.shape)
        #y1= F.conv2d(y_grouped, h1_theta, padding=1 ,groups=B*C)
        #y2 = y1.view(B, C, H, W)

        V_p = self.get_param(y)
        y1= torch.einsum("abcde,abf->bfde", (V_p, h2_theta.permute(2,0,1)))
        V_ps = self.get_param_spat(y1)
        # print(V_ps.shape)
        # print(h1_theta.shape)
        output= torch.einsum("abcde,abf->bfde", (V_ps, h1_theta.permute(2,0,1)))
        return output+y


    def get_param(self,x):
        R = x[:,:1,:,:]
        G = x[:,1:2,:,:]
        B = x[:,2:,:,:]
        C = torch.ones_like(R)
        return torch.stack([R, G, B, torch.pow(R,2), torch.pow(G,2), torch.pow(B,2), R * G, G * B, B * R, C])

    def get_param_spat(self,x):
        #R = x[:,:1,:,:]
        #G = x[:,1:2,:,:]
        #B = x[:,2:,:,:]
        R= torch.roll(x, shifts=1, dims=3)
        R[:,:,:, 0] = 0
        L= torch.roll(x, shifts=-1, dims=3)
        L[:,:,:, -1] = 0
        U= torch.roll(x, shifts=1, dims=2)
        U[:,:,0,:] = 0
        D= torch.roll(x, shifts=-1, dims=2)
        D[:,:,-1,:] = 0
        RU=torch.roll(torch.roll(x, shifts=1, dims=2), shifts=1, dims=3)
        RU[..., 0] = 0
        RU[:,:,0,:]=0
        RD=torch.roll(torch.roll(x, shifts=-1, dims=2), shifts=1, dims=3)
        RD[..., 0] = 0
        RD[:,:,-1,:] = 0
        LU=torch.roll(torch.roll(x, shifts=1, dims=2), shifts=-1, dims=3)
        LU[:,:,:, -1] = 0
        LU[:,:,0,:] = 0
        LD=torch.roll(torch.roll(x, shifts=-1, dims=2), shifts=-1, dims=3)
        LD[:,:,:, -1] = 0
        LD[:,:,-1,:] = 0
        C = torch.ones_like(R)
       # return torch.stack([R, G, B, torch.pow(R,2), torch.pow(G,2), torch.pow(B,2), R * G, G * B, B * R, C])
        #torch.stack([C,R,L,U,D,RD,RU,LU,LD,torch.pow(R,2),torch.pow(R,2),torch.pow(R,2),torch.pow(R,2)])
        Liniar=torch.stack([C,x,R,L,U,D,RD,RU,LU,LD])
        pow= torch.stack([torch.pow(x,2),torch.pow(R,2),torch.pow(L,2),torch.pow(U,2),torch.pow(D,2),torch.pow(RU,2),torch.pow(RD,2),torch.pow(LU,2),torch.pow(LD,2)])
        x_cross_term=torch.stack([x*R,x*L,x*U,x*D,x*RU,x*RD,x*LU,x*LD])
        R_cross_term=torch.stack([R*U,R*D,R*L,R*RU,R*RD,R*LU,R*LD])
        U_cross_term=torch.stack([U*D,U*L,U*RU,U*RD,U*LU,U*LD])
        D_cross_term=torch.stack([D*L,D*RU,D*RD,D*LU,D*LD])
        L_cross_term=torch.stack([L*RU,L*RD,L*LU,L*LD])
        RU_ct=torch.stack([RU*RD,RU*LU,RU*LD])
        RD_ct=torch.stack([RD*LU,RD*LD])
        LU_ct=torch.stack([LU*LD])

        return torch.cat([Liniar,pow,x_cross_term,R_cross_term,U_cross_term,D_cross_term,L_cross_term,RU_ct,RD_ct,LU_ct], dim=0)

    def val_loss(self,val_data,device):
      mse_loss = nn.MSELoss()

      model.eval()
      val_loss=0
      avg_loss=0
      with torch.no_grad():  # Disable gradient calculations
        for i,batch_data in enumerate(val_data):
            blur_img,resize_blur_img,clean_img= batch_data[0], batch_data[1], batch_data[2]
            blur_img,resize_blur_img,clean_img = blur_img.to(device) ,resize_blur_img.to(device), clean_img.to(device)
            blur_img,resize_blur_img,clean_img = normalize(blur_img,resize_blur_img,clean_img)

            # Forward pass
            res = model(resize_blur_img.permute(0, 2, 1, 3),blur_img.permute(0, 2, 1, 3))
            #loss = criterion(res, clean_img.permute(0, 2, 1, 3))
            loss = mse_loss(res, clean_img.permute(0, 2, 1, 3))

            val_loss += loss.item()
            avg_loss=val_loss/(i+1)

      model.train()
      return  avg_loss
    

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
        
#       loss+=self.criterion(self.transform[1](img_input),self.transform[1](img_target))
        
        loss+=self.huber(self.transform[1](img_input),self.transform[1](img_target))
        return loss#+self.psnr(img_input,img_target)+self.ssim(img_input,img_target)
    
def denormalize(img):
    mean = 127.5
    std = 127.5
    return img * std + mean  # Bring values back to [0, 255]
    
def normalize(img, real, targ):
    # Ensure correct shape: (B, C, H, W)
    if img.shape[-1] == 3:  # If NHWC, convert to NCHW
        img = img.permute(0, 3, 1, 2)
        real = real.permute(0, 3, 1, 2)
        targ = targ.permute(0, 3, 1, 2)

    # Standard normalization (assuming images in range [0,255])
    mean = 0.5 * 255  # Scalar
    std = 0.5 * 255  # Scalar

    img = (img - mean) / std
    real = (real - mean) / std
    targ = (targ - mean) / std

    return img, real, targ

# def normalize(img,real,targ):
#     img = img.permute(0,3,1,2)
#     real = real.permute(0,3,1,2)
#     targ = targ.permute(0,3,1,2)
#     mean_512 = torch.ones_like(img)*0.5*255
#     std_512 = torch.ones_like(img)*0.5*255
#     mean_256 = torch.ones_like(real)*0.5*255
#     std_256 = torch.ones_like(real)*0.5*255

#     img = (img-mean_512)/std_512
#     real = (real-mean_256)/std_256
#     targ = (targ-mean_512)/std_512

#     return img,real,targ
    

def main():
    """
    Use a main function to avoid running the training code when importing parts
    from this file (for example, test_model.py import the GoPro class).
    """
    # Define hyper parameters.
    n_epochs = 10
    decay_epoch = 30
    batch_size = 20
    lr = 9e-4
    epoch = 0 # index
    LossVec=[]

    validation_loss = 0
    best_val_loss = float("inf")
    best_model = None

    # Get the data
    dataset_path = "/mnt/d/datasets/DBlur/Gopro/"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for training efficiency
        transforms.ToTensor()
    ])
    full_train_dataset = GoProDataset(dataset_path, split='train', transform=transform)

    # Define split sizes (e.g., 80% train, 20% validation).
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split dataset (creating validation dataset).
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Define DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    print(f"Train size: {train_size}, Validation size: {val_size}")

    """
    Code that takes the first pair of images from the train dataset and displays it.
    Used it to make sure everything worked as expected, bu t we don't need it anymore.
    """
    # # Load a sample batch from the train dataset
    # sample_blur, sample_sharp = next(iter(train_loader))
    # print(sample_blur.shape, sample_sharp.shape)  # Expect (batch_size, 3, 256, 256)
    # # Convert PyTorch tensor to NumPy
    # blur_img = sample_blur[0].cpu().permute(1, 2, 0).numpy()
    # sharp_img = sample_sharp[0].cpu().permute(1, 2, 0).numpy()
    # # Convert from float range [0,1] to uint8 [0,255]
    # blur_img = (blur_img * 255).astype(np.uint8)
    # sharp_img = (sharp_img * 255).astype(np.uint8)
    # # Convert RGB format to BGR (required by cv2)
    # blur_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR)
    # sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_RGB2BGR)
    # # Show images
    # cv2.imshow("Blurred Image", blur_img)
    # cv2.imshow("Sharp Image", sharp_img)
    # cv2.waitKey(0)  # Wait until a key is pressed
    # cv2.destroyAllWindows()

    # Create model instance.
    model = Transformer(n_batch=batch_size).cuda()
    criterion = Loss()

    # TODO: What is this? is it required???
    lambda1 = lambda epoch: 0.55 ** (epoch) 

    optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # for i, (src_img, resized_img, target_img) in enumerate(train_loader):
    #     print(f"Batch {i}: {src_img.shape}, {resized_img.shape}, {target_img.shape}")


    # Train the model
    model.train()
    for epoch in range(0, n_epochs):
        torch.cuda.empty_cache()
        avg_loss = 0
        epoch_start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            start = time.time()
            blur_img,resize_blur_img,clean_img = batch_data[0], batch_data[1], batch_data[2]

            blur_img=blur_img.to(device)
            resize_blur_img=resize_blur_img.to(device)
            clean_img=clean_img.to(device)

            blur_img,resize_blur_img,clean_img = normalize(blur_img,resize_blur_img,clean_img)

            optimizer.zero_grad()

            #Inverse identity
            res = model(resize_blur_img.permute(0, 2, 1, 3),blur_img.permute(0, 2, 1, 3)) # previously for res 3, mask,src
            loss = criterion(res,clean_img.permute(0, 2, 1, 3))
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
                pic = (torch.cat([resize_blur_img,res, clean_img], dim=0).data + 1) / 2.0
                save_dir = "./results"
                torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, 200), nrow=3)


        # NOTE: Originally it was done every 5 epochs, but I don't see a reason to not do it each time (the only issue is its time consuming).
        # Calculate validation loss.
        model.eval()
        validation_loss = model.val_loss(val_loader,device)
        model.train()
        PSNR = 10*np.log10(1 / np.sqrt(validation_loss))

        # Save log
        with open("/mnt/d/deblurring/logs/losses.txt", "a") as f:
            f.write('-' * 89 + "\n")
            f.write('| end of epoch {:3d} | time: {:5.2f}s | '
                    'Loss {:5.4f} | Average loss {:5.4f} | PSNR loss {:5.4f} |'
                    ' Validation loss: {:5.4f} \n'.format(epoch, (time.time() - epoch_start_time), loss.item(), avg_loss, PSNR, validation_loss))
            f.write('-' * 89 + "\n")

        # Also print, for convenience.
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'Loss {:5.4f} | Average loss {:5.4f} | PSNR loss {:5.4f} |'
            ' Validation loss: {:5.4f} \n'.format(epoch, (time.time() - epoch_start_time), loss.item(), avg_loss, PSNR, validation_loss))
        print('-' * 89)

        # Save the best model.
        if validation_loss < best_val_loss:
            # TODO: This is a debug message, remove it later.
            print("Improvement !!!!!")
            print("Best validation loss was {0}, but now its {1}".format(best_val_loss, validation_loss))
            best_val_loss = validation_loss
            best_model = model

        # Save the current trained model.
        torch.save(model.state_dict(), '/mnt/d/deblurring/models/model_epoch_{0}.pth'.format(epoch))
        # NOTE: Load with "model.load_state_dict(torch.load("/path/to/mode/file"))"

        LossVec.append(loss.item())

        # Update learning rate.
        if (epoch+1)%30==0 and epoch < 320:
            lr_scheduler.step()

    # After all training, save the final best mode.
    torch.save(best_model.state_dict(), '/mnt/d/deblurring/models/best_model.pth'.format(epoch))
    print("Best model validation loss: {}".format(best_val_loss))

if __name__ == '__main__':
    main()





