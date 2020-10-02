import os
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DataModule(pl.LightningModule):

    #    this transform was used in the original implementation on TensorFlow
    class CustomNormalize(object):
        def __call__(self, img):
            img = img - 0.5
            img = img * 2.0
            return img

    class _Dataset(Dataset):
        def __init__(self, csv_data, im_folder, transforms):
            self.transforms = transforms
            self.im_folder = im_folder
            self.data = csv_data

        def __getitem__(self, item):
            line = self.data.iloc[item]
            image_path, labels = line[0], torch.tensor(line[1:])

            image_path = os.path.join(self.im_folder, image_path)

            image = Image.open(image_path).convert('RGB')

            if self.transforms is not None:
                image = self.transforms(image)

            return image, labels

        def __len__(self):
            return self.data.shape[0]

    def __init__(self, config):
        super().__init__()
        
        self.image_dir = config['image_dir']
        self.data_path = config['image_label_dict']
        self.batch_size = config['batch_size']
        
        print(
            f"Image dir: {self.image_dir}\n"
            f"Data path: {self.data_path}\n"
            f"Batch size: {self.batch_size}"
        )

        self.transforms = transforms.Compose([
            # Input PIL Image
            transforms.CenterCrop(150),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            DataModule.CustomNormalize()
        ])

        data = pd.read_csv(self.data_path)

        train_data, val_data = train_test_split(data, test_size=0.33, random_state=4)

        self.train_dataset = DataModule._Dataset(train_data, self.image_dir, self.transforms)
        self.val_dataset = DataModule._Dataset(val_data, self.image_dir, self.transforms)

        assert os.path.isdir(self.image_dir), f"Check image path:{self.image_dir}!"
        assert os.path.isfile(self.data_path), f"File {self.data_path} is not found!"

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False)
