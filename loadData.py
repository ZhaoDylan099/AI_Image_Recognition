from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


def load_data(data_dir, batch_size=32, shuffle=True, split = 0.8, img_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset = DataLoader(train_split, batch_size=batch_size, shuffle=shuffle)
    val_dataset = DataLoader(val_split, batch_size=batch_size, shuffle=shuffle)

    return train_dataset, val_dataset
        