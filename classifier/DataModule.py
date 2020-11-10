import os
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DataModule(pl.LightningModule):

    class CustomDataset(Dataset):
        def __init__(self, csv_data, im_folder, _transforms, from_explainer):
            self.transforms = _transforms
            self.im_folder = im_folder
            self.data = csv_data
            self.from_explainer = from_explainer

        def __getitem__(self, item):
            line = self.data.iloc[item]

            if not self.from_explainer:
                image_path, labels = line[0], torch.tensor(line[1:])
                image_path = os.path.join(self.im_folder, image_path)
            else:
                image_path, labels = line[0], torch.tensor(line[1:])[0]

            image = Image.open(image_path).convert('RGB')

            if self.transforms is not None:
                image = self.transforms(image)

            return image, labels

        def __len__(self):
            return self.data.shape[0]

    def __init__(self, config, from_explainer=False):
        super().__init__()

        self.image_dir = './data/CelebA/images/'
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
            # The same, but parallelize
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0])
        ])

        if not from_explainer:
            self.data = pd.read_csv(self.data_path)
        else:
            attr_names, attr_list = self.read_data_file(self.data_path)
            self.data = pd.DataFrame(attr_list.items(), columns=['Path', 'Bin'])

        train_data, val_data = train_test_split(self.data, test_size=0.33, random_state=4)
        train_data.index = range(len(train_data))
        val_data.index = range(len(val_data))

        self.train_dataset = DataModule.CustomDataset(train_data, self.image_dir, self.transforms, from_explainer)
        self.val_dataset = DataModule.CustomDataset(val_data, self.image_dir, self.transforms, from_explainer)

        assert os.path.isdir(self.image_dir), f"Check image path:{self.image_dir}!"
        assert os.path.isfile(self.data_path), f"File {self.data_path} is not found!"

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True, num_workers=32, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False, num_workers=32, drop_last=True)

    @staticmethod
    def read_data_file(file_path, image_dir=''):
        attr_list = {}
        path = file_path
        file = open(path, 'r')
        n = file.readline()
        n = int(n.split('\n')[0])  # Number of images
        attr_line = file.readline()
        attr_names = attr_line.split('\n')[0].split()  # attribute name
        for line in file:
            row = line.split('\n')[0].split()
            img_name = os.path.join(image_dir, row.pop(0))
            try:
                row = [float(val) for val in row]
            except:
                print(line)
                img_name = img_name + ' ' + str(row[0])
                row.pop(0)
                row = [float(val) for val in row]

            attr_list[img_name] = row

        file.close()

        return attr_names, attr_list
