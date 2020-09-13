# Description: read data from configs and preprocess it

import os

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Think about
# output_dir = os.path.join(config['log_dir'], config['name'])

# Read data file = df.iloc: popping attrubutes by their name

# Убрать .npy файлы!

class DataModule(pl.LightningModule):
    class _Dataset(Dataset):
        def __init__(self, csv_data, im_folder, transforms, shuffle):
            self.transforms = transforms
            self.im_folder = im_folder

            self.data = csv_data
            if shuffle:
                self.data = self.data.sample(frac=1)

        def __getitem__(self, item):
            # TODO(check outputting labels as 1D tensors)
            line = self.data.iloc[item]
            image_path, labels = line[0], line[1:]

            image_path = os.path.join(self.im_folder, image_path)

            image = Image.open(image_path).convert('RGB')

            if self.transforms is not None:
                image = self.transforms(image)

            return image, torch.tensor(labels)

        def __len__(self):
            return self.data.shape[0]

    def __init__(self, config_path='configs/celebA_YSBBB_Classifier.yaml'):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        print("Config  file:", self.config, sep='\n')

        self.image_dir = self.config['image_dir']
        self.data_path = self.config['image_label_dict']
        self.batch_size = self.config['batch_size']
        print(
            f"Image dir: {self.image_dir}\n"
            f"Data path: {self.data_path}\n"
            f"Batch size: {self.batch_size}"
        )

        self.transforms = transforms.Compose([
            # Input PIL Image
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=[2, 2, 2])
        ])

        assert os.path.isdir(self.image_dir), f"Check image path:{self.image_dir}!"
        assert os.path.isfile(self.data_path), f"File {self.data_path} is not found!"

    # On one CPU, not paralleled
    def prepare_data(self):
        data = pd.read_csv(self.data_path)

        # They don't have test data, only val data, why?
        train_data, val_data = train_test_split(data, test_size=0.33)

        self.train_dataset = DataModule._Dataset(train_data, self.image_dir, self.transforms, True)
        self.val_dataset = DataModule._Dataset(val_data, self.image_dir, self.transforms, False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False)


if __name__ == '__main__':
    dataModule = DataModule()
    dataModule.prepare_data()
    print(len(dataModule.train_dataset), len(dataModule.val_dataset))

    # for i in range(len(dataModule.train_dataset)):
    #     print(i, ":", dataModule.train_dataset[i][1])
    #
    # size = len(dataModule.train_dataset)
    # for i in range(len(dataModule.val_dataset)):
    #     print(i + size, ":", dataModule.val_dataset[i][1])

    import matplotlib.pyplot as plt
    plt.imshow(dataModule.train_dataset[1][0][0].numpy(), cmap='gray')
    plt.show()
