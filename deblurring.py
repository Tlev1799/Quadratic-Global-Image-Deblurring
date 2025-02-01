import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class GoProDataset(Dataset):
    """
    We are using the GoPro dataset because it has generic images,
    instead of focusing on pedestrians or faces or text.
    The downside is this dataset is split to test/train without validation,
    so we'll need to do it manually.    
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.blur_dir = os.path.join(root_dir, split, 'blur')
        self.sharp_dir = os.path.join(root_dir, split, 'sharp')
        self.image_filenames = sorted(os.listdir(self.blur_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.image_filenames[idx])
        sharp_path = os.path.join(self.sharp_dir, self.image_filenames[idx])

        blur_image = Image.open(blur_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image

# Example usage
dataset_path = "C:\\Users\\user\\GitRepos\\Single-Image-Super-Resolution\\datasets\\GoPro"
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for training efficiency
    transforms.ToTensor()
])

train_dataset = GoProDataset(dataset_path, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load a sample batch
sample_blur, sample_sharp = next(iter(train_loader))
print(sample_blur.shape, sample_sharp.shape)  # Expect (batch_size, 3, 256, 256)