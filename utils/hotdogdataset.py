import os
import zipfile
import requests
import hashlib
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import math
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import torch
import numpy as np

class DLHotDogDataset(Dataset):
    def __init__(self, root: Union[str, Path], url="https://www.doc.ic.ac.uk/~bkainz/teaching/DL/hot-dog-10k.zip",  
                 md5_url="https://www.doc.ic.ac.uk/~bkainz/teaching/DL/hot-dog-10k.md5", 
                 transform: Optional[Callable] =None, split='train', preload=False):
        self.url = url
        self.root = Path(root)
        self.md5_url = md5_url
        self.zip_filename = os.path.join(root, "hot-dog-10k.zip")
        self.md5_filename = os.path.join(root, "hot-dog-10k.md5")
        self.data_folder = Path(os.path.join(root, "hot-dog-10k"))
        self.transform = transform
        self.preload = preload

        if not os.path.exists(self.data_folder):
            self.download_and_extract()
            self.rename_files()

        self.image_paths = list(self.data_folder.glob('**/*.jpg')) + list(self.data_folder.glob('**/*.png')) + list(self.data_folder.glob('**/*.jpeg'))
        slen = len(self.image_paths)
        if split == 'train':
            self.image_paths = self.image_paths[:math.floor(slen*0.9)]
        elif split == 'test':
            self.image_paths = self.image_paths[math.floor(slen*0.9):]
        else:
            print("split type unknown, using all data")
        print("number of hot dogs: ", len(self.image_paths))

        if self.preload:
            self.images = [self.load_image(path) for path in self.image_paths]

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def download_file(self, url, filename):
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)

    def verify_md5(self, filename, md5_filename):
        with open(md5_filename, 'r') as f:
            expected_md5_line = f.read().strip()
            expected_md5 = expected_md5_line.split()[0]
            print(f"Expected MD5: {expected_md5}")

        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_md5 = hash_md5.hexdigest()
        print(f"Computed MD5: {file_md5}")

        return file_md5 == expected_md5

    def download_and_extract(self):
        os.makedirs(self.root, exist_ok=True)

        # Download the ZIP file if it doesn't exist
        if not os.path.exists(self.zip_filename):
            self.download_file(self.url, self.zip_filename)

        # Download the MD5 file if URL is provided
        if self.md5_url:
            if not os.path.exists(self.md5_filename):
                self.download_file(self.md5_url, self.md5_filename)

             # Verify the integrity of the ZIP file
            is_md5_valid = self.verify_md5(self.zip_filename, self.md5_filename)
            print(f"MD5 verification result: {is_md5_valid}")
            if not is_md5_valid:
                raise ValueError("MD5 checksum does not match. The file might be corrupted.")

        # Extract the ZIP file
        with zipfile.ZipFile(self.zip_filename, 'r') as zip_ref:
            zip_ref.extractall(self.data_folder)
    
    def rename_files(self):
        print(f"Renaming files in {self.data_folder}...")
        for root, _, files in os.walk(self.data_folder):
            for i, filename in enumerate(sorted(files), start=1):
                file_extension = os.path.splitext(filename)[1]
                new_filename = f"{i:08d}{file_extension}"
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
        print("Renaming complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
        else:
            img_path = self.image_paths[idx]
            image = self.load_image(img_path)
        return image, 0

    def calculate_mean_and_std(self):
        if self.preload:
            images = self.images
        else:
            images = [self.load_image(path) for path in self.image_paths]

        # Convert images to tensors and concatenate them
        all_data = torch.cat([transforms.ToTensor()(np.array(image)).view(-1, 3) for image in images], dim=0)
        
        mean = all_data.mean()
        std = all_data.std()
        
        return mean, std
