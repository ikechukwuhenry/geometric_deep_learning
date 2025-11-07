import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Custom transform to apply random rotation and reflection
so2o2_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 360)),  # SO(2) rotation
    transforms.RandomHorizontalFlip(),        # Reflection
    transforms.RandomVerticalFlip(),          # Reflection
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=so2o2_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load test set (no augmentation, just tensor conversion)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Example: visualize  one batch of augmented images
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(12, 2))

for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')
    axes[i].set_title(f'Label: {labels[i].item()}')
    axes[i].axis('off')

plt.suptitle('MNIST with Random SO(2) Rotation and Reflection Augmentation', fontsize=14)
plt.show()