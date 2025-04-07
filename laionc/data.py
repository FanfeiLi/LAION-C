import os
from torchvision import transforms
from torch.utils.data import DataLoader
from .Imagenetc_class import ImageNetDataset
import os
import requests
import zipfile
import gdown

def download_dataset(location="default_location", augmentation_type=None, intensity_level=1):
    """
    Downloads only the specified augmentation type and intensity level.
    
    Args:
        location (str): Root directory to save the dataset.
        augmentation_type (str): Type of augmentation (e.g., 'brightness', 'rotation', 'blur').
        intensity_level (int): Intensity level of the augmentation (e.g., 1, 2, 3).
    """
    # Construct the subfolder path based on augmentation type and intensity level
    subfolder = os.path.join(location, augmentation_type)
    
    # Skip download if the data already exists
    if os.path.exists(subfolder):
        print(f"{subfolder} already exists. Skipping download.")
        return
    
    # Define download URL for specific augmentation and intensity (modify with actual URL)
    url = f"https://zenodo.org/api/records/14051887/draft/files/{augmentation_type}.zip/content"
    # Ensure the download directory exists
    if not os.path.exists(location):
        os.makedirs(location)
    
    # Download and extract only the necessary folder
    zip_path = os.path.join(location, f"{augmentation_type}.zip")
    
    print(f"Downloading {augmentation_type} with intensity {intensity_level} to {zip_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # Extract the zip file and clean up
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(location)
        os.remove(zip_path)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Failed to extract dataset: {e}")


def get_dataloader(batch_size, num_workers, dataset_location, transform=None):
    transform = transform or transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageNetDataset(location=dataset_location, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
