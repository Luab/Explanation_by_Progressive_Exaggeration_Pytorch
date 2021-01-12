import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class DataModule(pl.LightningModule):
    class CustomDataset(Dataset):
        def __init__(self, df, image_dir, transforms, to_explainer, append_to_path=''):
            self.transforms = transforms
            self.image_dir = image_dir
            self.df = df
            self.to_explainer = to_explainer
            self.append_to_path = append_to_path

        def __getitem__(self, item):
            line = self.df.iloc[item]
            
            if self.to_explainer:
                image_path = self.append_to_path + line[0]
            else:
                image_path = os.path.join(self.image_dir, line[0])
            
            labels = torch.tensor(line[1:])
            image = Image.open(image_path).convert('RGB')

            if self.transforms is not None:
                image = self.transforms(image)

            return image, labels

        def __len__(self):
            return self.df.shape[0]

    def __init__(self, config, to_explainer=False, append_to_path=''):
        super().__init__()

        self.batch_size = config['batch_size']
        
        data = None
        if not to_explainer:
            data = pd.read_csv(config['image_label_dict'])
        else:
            attr_names, attr_list = self.read_data_file(file_path=config['image_label_dict'])
            data = pd.DataFrame(attr_list.items(), columns=['Path', 'Bin'])

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(150),
            torchvision.transforms.Resize(size=(128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0])
        ])
        
        train_data, val_data = train_test_split(data, test_size=0.33, random_state=4)
        train_data.index = range(len(train_data))
        val_data.index = range(len(val_data))

        self.train_dataset = DataModule.CustomDataset(df=train_data,
                                                      image_dir='./data/CelebA/images/',
                                                      transforms=transforms,
                                                      to_explainer=to_explainer,
                                                      append_to_path=append_to_path)
        self.val_dataset = DataModule.CustomDataset(df=val_data,
                                                    image_dir='./data/CelebA/images/',
                                                    transforms=transforms,
                                                    to_explainer=to_explainer,
                                                    append_to_path=append_to_path)

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
