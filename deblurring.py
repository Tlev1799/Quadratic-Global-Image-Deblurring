import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import kornia

from utils import rgb_to_hsv, rgb_to_lab, rgb_to_grayscale, calculate_ssim_torch, calculate_psnr_torch, GoProDataset


import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define the model and loss functions
# class Transformer(nn.Module):
#     def __init__(self, Norm=nn.InstanceNorm2d, Act=nn.ReLU(True), in_features=64, in_channels=3):
#         super(Transformer, self).__init__()

#         # Sequential layers for the down-sampled image (256x256) to output a (82x3) transformation vector
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels, in_features, kernel_size=5, stride=2, padding=2),
#             Act,
#             nn.Conv2d(in_features, in_features * 2, kernel_size=3, stride=2, padding=1),
#             Norm(in_features * 2),
#             Act,
#             nn.AdaptiveAvgPool2d((6, 6)),
#             nn.Flatten(),
#             nn.Linear(in_features * 2 * 6 * 6, 128),
#             Act,
#             nn.Linear(128, 246),
#             nn.Unflatten(1, (82, 3))  # Output shape: (batch, 82, 3)
#         )

#     def forward(self, x, y):

#         """
#         x - Blurred image original size.
#         y - Blurred image downsampled to 256x256.
#         """

#         # Convert RGB to HSV
#         x_hsv = kornia.color.rgb_to_hsv(x)
#         y_hsv = kornia.color.rgb_to_hsv(y)

#         # Separate the HSV channels (Hue, Saturation, Value)
#         x_h, x_s, x_v = x_hsv[:, 0:1, :, :], x_hsv[:, 1:2, :, :], x_hsv[:, 2:3, :, :]
#         y_h, y_s, y_v = y_hsv[:, 0:1, :, :], y_hsv[:, 1:2, :, :], y_hsv[:, 2:3, :, :]


#         # Get transformation parameters from down-sampled image y
#         h_theta = self.feature_extractor(y_hsv)  # Shape: (batch, 82, 3)

#         # Compute basis values for each pixel in the original image x
#         V_p = self.get_param(x_hsv)  # Shape: (batch, 82, H, W)

#         # Apply transformation: pixel-wise matrix multiplication
#         # V_p of shape (20, 82, 256, 256)
#         # h_theta of shape (20, 82, 3)

#         output_hsv = torch.einsum('bijk,bil->bljk', V_p, h_theta)
#         output_rgb = kornia.color.hsv_to_rgb(output_hsv)

#         return output_rgb # shape (20, 3, 256, 256)

#     def get_param(self, x):
#         """Computes quadratic basis values for each pixel and its 8 neighbors."""
#         R = x[:, :1, :, :]  # Red channel
#         G = x[:, 1:2, :, :]  # Green channel
#         B = x[:, 2:, :, :]  # Blue channel
#         C = torch.ones_like(R)  # Constant term

#         # Compute quadratic basis terms (9 spatial values per channel)
#         R2 = torch.pow(R, 2)
#         G2 = torch.pow(G, 2)
#         B2 = torch.pow(B, 2)
#         RG = R * G
#         GB = G * B
#         BR = B * R

#         # Stack to create 9 feature channels
#         basis = torch.cat([R, G, B, R2, G2, B2, RG, GB, BR], dim=1)  # Shape: (batch, 10, H, W)

#         # Use padding to ensure each pixel has 8 neighbors
#         basis_padded = F.pad(basis, (1, 1, 1, 1))  # Padding with zeros.

#         # Extract 9 spatial values for each channel (9 neighbors including the pixel itself)
#         basis_stacked = torch.cat([ 
#             basis_padded[:, :, :-2, :-2], basis_padded[:, :, :-2, 1:-1], basis_padded[:, :, :-2, 2:],  # Top row
#             basis_padded[:, :, 1:-1, :-2], basis_padded[:, :, 1:-1, 1:-1], basis_padded[:, :, 1:-1, 2:],  # Middle row
#             basis_padded[:, :, 2:, :-2], basis_padded[:, :, 2:, 1:-1], basis_padded[:, :, 2:, 2:]   # Bottom row
#         ], dim=1)  # Shape: (batch, 81, H, W)
#         basis_stacked = torch.cat([basis_stacked, C], dim=1) # Shape: (batch, 82, H, W)
        
#         return basis_stacked


#     def val_loss(self, val_data, loss_fn, device):
#         self.eval()

#         val_loss = 0
#         val_PSNR_loss = 0
#         val_SSIM_loss = 0
#         num_batches = 0

#         with torch.no_grad():  # Disable gradient calculations
#             for batch_data in val_data:
#                 blur_img, resize_blur_img, clean_img = (
#                     batch_data[0].to(device),
#                     batch_data[1].to(device),
#                     batch_data[2].to(device),
#                 )

#                 # Forward pass
#                 res = self(resize_blur_img, blur_img)  # Ensure correct input order

#                 loss, PSNR_loss, SSIM_loss = loss_fn(res, clean_img, is_validation=True)

#                 val_loss += loss.item()
#                 val_PSNR_loss += PSNR_loss.item()
#                 val_SSIM_loss += SSIM_loss.item()
#                 num_batches += 1

#         self.train()

#         # Avoid division by zero
#         if num_batches > 0:
#             val_loss /= num_batches
#             val_PSNR_loss /= num_batches
#             val_SSIM_loss /= num_batches

#         return val_loss, val_PSNR_loss, val_SSIM_loss

class Transformer(nn.Module):
    def __init__(self, Norm=nn.InstanceNorm2d, Act=nn.ReLU(True), in_features=512, in_channels=2):
        super(Transformer, self).__init__()

        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_features // 4, kernel_size=5, stride=2, padding=2),  # (2 -> 128)
            Act,
            nn.Conv2d(in_features // 4, in_features // 2, kernel_size=3, stride=2, padding=1),  # (128 -> 256)
            Norm(in_features // 2),
            Act,
            nn.Conv2d(in_features // 2, in_features, kernel_size=3, stride=2, padding=1),  # (256 -> 512)
            Norm(in_features),
            Act,
            nn.AdaptiveAvgPool2d((6, 6)),  # Output size: (512, 6, 6)
            nn.Flatten()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features * 6 * 6, 4096),  # (512*6*6 -> 4096)
            Act,
            nn.Linear(4096, 1024),  # (4096 -> 1024)
            Act,
            nn.Linear(1024, 92),  # (1024 -> 92)
            nn.Unflatten(1, (46, 2))  # Output shape: (batch, 46, 2)
        )

    def forward(self, x, y):
        """
        x - Blurred image original size.
        y - Blurred image downsampled to 256x256.
        """
        # Convert x and y to HSV color space
        x_hsv = kornia.color.rgb_to_hsv(x)
        y_hsv = kornia.color.rgb_to_hsv(y)

        # Separate the HSV channels (Hue, Saturation, Value)
        x_h, x_s, x_v = x_hsv[:, 0:1, :, :], x_hsv[:, 1:2, :, :], x_hsv[:, 2:3, :, :]
        y_h, y_s, y_v = y_hsv[:, 0:1, :, :], y_hsv[:, 1:2, :, :], y_hsv[:, 2:3, :, :]

        #import pdb;pdb.set_trace()

        # Keep the Hue (H) channels as they are
        # Get transformation parameters from the down-sampled image y (using the S and V channels)
        h_theta = self.fc(self.feature_extractor(torch.cat([y_s, y_v], dim=1)))  # Shape: (batch, 46, 2)

        # Add a column of 1's to h_theta to ensure H channel remains unchanged
        ones = torch.ones_like(h_theta[:, :, :1])  # Shape: (batch, 46, 1)
        h_theta = torch.cat([ones, h_theta], dim=2)  # Shape: (batch, 46, 3)

        # Compute basis values for each pixel in the original image x (using the S and V channels)
        SV_p = self.get_param(torch.cat([x_s, x_v], dim=1))  # Shape: (batch, 46, H, W)

        # Apply transformation: pixel-wise matrix multiplication
        output_v = torch.einsum('bijk,bil->bljk', SV_p, h_theta)

        # Reconstruct the image in HSV color space with the transformed S and V channels
        output_hsv = torch.cat([x_h, output_v[:, 1:2, :, :], output_v[:, 2:3, :, :]], dim=1)  # Shape: (batch, 3, H, W)

        # Convert the result back to RGB
        output_rgb = kornia.color.hsv_to_rgb(output_hsv)

        return output_rgb+y  # Shape: (batch, 3, H, W)

    def get_param(self, x):
        """Computes quadratic basis values for each pixel and its 8 neighbors, for only S and V channels."""
        S = x[:, :1, :, :]  # Saturation channel
        V = x[:, 1:2, :, :]  # Value channel
        C = torch.ones_like(S)  # Constant term

        # Compute quadratic basis terms (5 spatial values per channel for each pixel and its 8 neighbors)
        S2 = torch.pow(S, 2)
        V2 = torch.pow(V, 2)
        SV = S * V

        # Stack to create 5 feature channels for each neighbor
        basis = torch.cat([S, V, S2, V2, SV], dim=1)  # Shape: (batch, 5, H, W)

        # Use padding to ensure each pixel has 8 neighbors
        basis_padded = F.pad(basis, (1, 1, 1, 1))  # Padding with zeros.

        # Extract 9 spatial values for each channel (9 neighbors including the pixel itself)
        basis_stacked = torch.cat([ 
            basis_padded[:, :, :-2, :-2], basis_padded[:, :, :-2, 1:-1], basis_padded[:, :, :-2, 2:],  # Top row
            basis_padded[:, :, 1:-1, :-2], basis_padded[:, :, 1:-1, 1:-1], basis_padded[:, :, 1:-1, 2:],  # Middle row
            basis_padded[:, :, 2:, :-2], basis_padded[:, :, 2:, 1:-1], basis_padded[:, :, 2:, 2:]   # Bottom row
        ], dim=1)  # Shape: (batch, 45, H, W)
        basis_stacked = torch.cat([basis_stacked, C], dim=1)  # Add constant term to the stack, Shape: (batch, 46, H, W)
        
        return basis_stacked

    def val_loss(self, val_data, loss_fn, device):
        self.eval()

        val_loss = 0
        val_PSNR_loss = 0
        val_SSIM_loss = 0
        num_batches = 0

        with torch.no_grad():  # Disable gradient calculations
            for batch_data in val_data:
                blur_img, resize_blur_img, clean_img = (
                    batch_data[0].to(device),
                    batch_data[1].to(device),
                    batch_data[2].to(device),
                )

                # Forward pass
                res = self(resize_blur_img, blur_img)  # Ensure correct input order

                loss, PSNR_loss, SSIM_loss = loss_fn(res, clean_img, is_validation=True)

                val_loss += loss.item()
                val_PSNR_loss += PSNR_loss.item()
                val_SSIM_loss += SSIM_loss.item()
                num_batches += 1

        self.train()

        # Avoid division by zero
        if num_batches > 0:
            val_loss /= num_batches
            val_PSNR_loss /= num_batches
            val_SSIM_loss /= num_batches

        return val_loss, val_PSNR_loss, val_SSIM_loss


    def val_loss(self, val_data, loss_fn, device):
        self.eval()

        val_loss = 0
        val_PSNR_loss = 0
        val_SSIM_loss = 0
        num_batches = 0

        with torch.no_grad():  # Disable gradient calculations
            for batch_data in val_data:
                blur_img, resize_blur_img, clean_img = (
                    batch_data[0].to(device),
                    batch_data[1].to(device),
                    batch_data[2].to(device),
                )

                # Forward pass
                res = self(resize_blur_img, blur_img)  # Ensure correct input order

                loss, PSNR_loss, SSIM_loss = loss_fn(res, clean_img, is_validation=True)

                val_loss += loss.item()
                val_PSNR_loss += PSNR_loss.item()
                val_SSIM_loss += SSIM_loss.item()
                num_batches += 1

        self.train()

        # Avoid division by zero
        if num_batches > 0:
            val_loss /= num_batches
            val_PSNR_loss /= num_batches
            val_SSIM_loss /= num_batches

        return val_loss, val_PSNR_loss, val_SSIM_loss



class Loss():
    def __init__(self, col_hsv=False, col_lab=True, col_gray=True, 
                 huber_weight=0, mse_weight=1, perceptual_weight=0.05, device='cuda'):
        # Color transformation options
        self.transform = list()
        if col_hsv:
            self.transform.append(rgb_to_hsv)
        if col_lab:
            print("lab")
            self.transform.append(rgb_to_lab)
        if col_gray:
            print("gray")
            self.transform.append(rgb_to_grayscale)

        # Loss functions
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.huber_weight = huber_weight

        self.criterion = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

        # Pre-trained VGG model for perceptual loss
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg = self.vgg.to(device=device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze the weights of VGG
        
        # PSNR and SSIM from Kornia
        #self.psnr = kornia.losses.PSNRLoss(1)

    def perceptual_loss(self, img_input, img_target):
        # Extract feature maps from VGG
        input_features = self.vgg(img_input)
        target_features = self.vgg(img_target)
        
        # Compute L2 loss on feature maps (Perceptual loss)
        loss = F.mse_loss(input_features, target_features)
        return loss

    def __call__(self, img_input, img_target, is_validation=False):
        loss = 0.0
        
        # # Apply color transformations
        # input = self.transform[0](img_input)
        # target = self.transform[0](img_target)
        input = img_input
        target = img_target
        
        # MSE loss for color space transformation
        mse_loss = self.criterion(input[:, 0, :, :], target[:, 0, :, :])
        mse_loss += self.criterion(input[:, 1, :, :], target[:, 1, :, :])
        mse_loss += self.criterion(input[:, 2, :, :], target[:, 2, :, :])
    
        loss += self.mse_weight * mse_loss
        
        # Huber loss for second transformation (e.g. LAB or grayscale)
        loss += self.huber_weight * self.huber(img_input, img_target)

        # Perceptual loss (only during training)
        if not is_validation:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(img_input, img_target)
            loss += perceptual_loss

        if is_validation:
            # For validation, compute PSNR and SSIM
            #psnr_value = self.psnr(img_input, img_target).mean()
            #import pdb; pdb.set_trace()
            psnr_value = calculate_psnr_torch(mse_loss)
            ssim_value = calculate_ssim_torch(img_input, img_target)
            return loss, psnr_value, ssim_value

        return loss
    
def main():
    # Define hyper parameters.
    n_epochs = 250
    batch_size = 16
    lr = 5e-5
    epoch = 0 # index

    validation_loss = 0
    psnr_loss = 0
    ssim_loss = 0

    best_val_loss = float("inf")
    best_val_model = None

    best_psnr_loss = 0
    best_psnr_model = None

    best_ssim_loss = -1
    best_ssim_model = None

    LossVec=[]
    PSNRVec=[]
    SSIMVec=[]

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
    print(f"Number of epochs we run: {n_epochs}")

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
    model = Transformer().cuda()
    criterion = Loss()

    lambda1 = lambda epoch: 0.55 ** (epoch)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # for i, (src_img, resized_img, target_img) in enumerate(train_loader):
    #     print(f"Batch {i}: {src_img.shape}, {resized_img.shape}, {target_img.shape}")


    # Train the model
    model.train()
    for epoch in range(0, n_epochs):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            #import pdb; pdb.set_trace()
            blur_img, resize_blur_img, clean_img = batch_data[0], batch_data[1], batch_data[2]

            blur_img = blur_img.to(device)
            resize_blur_img = resize_blur_img.to(device)
            clean_img = clean_img.to(device)

            #blur_img, resize_blur_img, clean_img = normalize(blur_img, resize_blur_img, clean_img)
            optimizer.zero_grad()

            res = model(resize_blur_img, blur_img) # previously for res 3, mask,src
            train_loss = criterion(res, clean_img, is_validation=False)
            train_loss.sum().backward()
            optimizer.step()

        # NOTE: Originally it was done every 5 epochs, but I don't see a reason to not do it each time (the only issue is its time consuming).
        # Calculate validation loss.
        model.eval()
        validation_loss, psnr_loss, ssim_loss = model.val_loss(val_loader, criterion, device)
        model.train()

        # Save log
        with open("/mnt/d/deblurring/logs/losses.txt", "a") as f:
            f.write('-' * 89 + "\n")
            f.write('| end of epoch {:3d} | time: {:5.2f}s | '
                    'MSE Loss {} | SSIM loss {} | PSNR loss {} \n'.format(
                        epoch, (time.time() - epoch_start_time), validation_loss, ssim_loss, psnr_loss))
            f.write('-' * 89 + "\n")

        # Also print, for convenience.
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'MSE Loss {} | SSIM loss {} | PSNR loss {}'.format(
                    epoch, (time.time() - epoch_start_time), validation_loss, ssim_loss, psnr_loss))
        print('-' * 89)

        # Save the best models.
        if validation_loss < best_val_loss:
            print("Best validation loss was {0}, but now its {1}".format(best_val_loss, validation_loss))
            best_val_loss = validation_loss
            best_val_model = model

        if psnr_loss > best_psnr_loss:
            print("Best psnr loss was {0}, but now its {1}".format(best_psnr_loss, psnr_loss))
            best_psnr_loss = psnr_loss
            best_psnr_model = model

        if ssim_loss > best_ssim_loss:
            print("Best ssim loss was {0}, but now its {1}".format(best_ssim_loss, ssim_loss))
            best_ssim_loss = ssim_loss
            best_ssim_model = model

        # Save the current trained model.
        torch.save(model.state_dict(), '/mnt/d/deblurring/models/model_epoch_{0}.pth'.format(epoch))
        # NOTE: Load with "model.load_state_dict(torch.load("/path/to/mode/file"))"

        LossVec.append(validation_loss)
        PSNRVec.append(psnr_loss)
        SSIMVec.append(ssim_loss)

        # Update learning rate.
        #if (epoch+1)%30==0 and epoch < 320:
        if (epoch+1)%10 == 0:
            lr_scheduler.step()

    # Save the final best model for each loss.
    torch.save(best_val_model.state_dict(), '/mnt/d/deblurring/models/best/best_val_model.pth')
    torch.save(best_psnr_model.state_dict(), '/mnt/d/deblurring/models/best/best_psnr_model.pth')
    torch.save(best_ssim_model.state_dict(), '/mnt/d/deblurring/models/best/best_ssim_model.pth')
    print("Best models losses:\nMSE & Huber: {}\npsnr: {}\nssim: {}".format(best_val_loss, best_psnr_loss, best_ssim_loss))

    # Save the loss vectors.
    torch.save(torch.tensor(LossVec), '/mnt/d/deblurring/losses/val_losses.pth')
    torch.save(torch.tensor(PSNRVec), '/mnt/d/deblurring/losses/psnr_losses.pth')
    torch.save(torch.tensor(SSIMVec), '/mnt/d/deblurring/losses/ssim_losses.pth')


if __name__ == '__main__':
    main()





