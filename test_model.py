from deblurring import GoProDataset, Transformer, normalize, denormalize
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import os

batch_size = 20

# Use GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Start")

# Get the test set
dataset_path = "/mnt/d/datasets/DBlur/Gopro/"
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for training efficiency.
    transforms.ToTensor()
])
test_dataset = GoProDataset(dataset_path, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=4, drop_last=True)

# Load trained model.
model = Transformer(n_batch=batch_size).cuda() # Replace with your actual model class
model.load_state_dict(torch.load("/mnt/d/deblurring/models/best_model"))
model.to(device)
model.eval()

print("Loaded model")

# Convert tensor to image
to_pil = transforms.ToPILImage()

# Plot images
fig, axes = plt.subplots(5, 3, figsize=(9, 15))

for i, batch_data in enumerate(test_loader):
    if i >= 5:
        break

    sample_dir = os.path.join("/mnt/d/deblurring/results/", f"sample_{i}")
    os.makedirs(sample_dir, exist_ok=True)

    blur_img, resize_blur_img, sharp_img = batch_data[0], batch_data[1], batch_data[2]

    blur_img = blur_img.to(device)
    resize_blur_img = resize_blur_img.to(device)
    sharp_img = sharp_img.to(device)


    blur_img, resize_blur_img, sharp_img = normalize(blur_img, resize_blur_img, sharp_img)
    print(blur_img.shape)

    with torch.no_grad():
        output = model(resize_blur_img, blur_img)

    print(f"Used model for sample {i}")

    blur_img = denormalize(blur_img)
    resize_blur_img = denormalize(resize_blur_img)
    output = denormalize(output)
    sharp_img = denormalize(sharp_img)

    print(blur_img.shape)

    save_image(blur_img[0].cpu(), os.path.join(sample_dir, "blurred.png"))
    save_image(output[0].cpu(), os.path.join(sample_dir, "predicted.png"))
    save_image(sharp_img[0].cpu(), os.path.join(sample_dir, "expected.png"))
