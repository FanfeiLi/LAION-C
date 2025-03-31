#from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import pathlib
import os
class ImageNetDataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}")
        self.transform = transform
        self.dataset = ImageFolder(self.dataset_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        img_path = self.dataset.imgs[i][0]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_path



