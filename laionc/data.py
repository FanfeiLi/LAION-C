import os
from torchvision import transforms
from torch.utils.data import DataLoader
from .Imagenetc_class import ImageNetDataset

def get_dataloader(batch_size, num_workers, dataset_location, transform=None):
    transform = transform or transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageNetDataset(location=dataset_location, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
