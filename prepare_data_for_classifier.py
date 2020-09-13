import os
import zipfile

import pandas as pd
import pytorch_lightning as pl


class DataModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.data_dir = "./data"
        self.celeba_dir = os.path.join(self.data_dir, "CelebA")
        self.zip_path = os.path.join(self.data_dir, "img_align_celeba.zip")
        self.image_dir = os.path.join(self.celeba_dir, "images")
        self.text_dir = os.path.join(self.celeba_dir, "list_attr_celeba.txt")

        assert (os.path.exists(self.zip_path), f"File img_align_celeba.zip does not lie in {self.celeba_dir} folder!")

    # On one CPU, not paralleled
    def prepare_data(self):

        # Checking that path images is not empty
        if os.path.exists(self.zip_save):
            if len(os.listdir(self.zip_save)) == 202599:
                print("Everything is okay, data is already preprocessed!")
                return
            else:
                print(
                    f"Something went wrong: found {len(os.listdir(self.zip_save))} files, deleting folder and trying again!")
                os.remove(self.zip_save)

        # Unpacking archive and moving it into './data/CelebA/images'
        with zipfile.ZipFile(self.zip_path) as zf:
            zf.extractall(self.celeba_dir)

        os.rename(os.path.join(self.celeba_dir, "img_align_celeba"), self.zip_save)

        print("Image dir:", self.image_dir)
        print("Text dir:", self.text_dir)

        data_df = pd.read_csv(self.text_dir)
        print(f"Short content of file {self.text_dir}", data_df.head(5), sep="\n", end="\n\n")

        print("Number of images:", data_df.shape[0])
        print("Categories:", data_df.columns)
        print("First attributes:", data_df.iloc[0])

        # Write the label file for target attribute binary classification
        attributes = "Young"
        bin_dataframe = data_df[["Path", attributes]]
        bin_filename = attributes + "_binary_classification.txt"
        bin_dataframe.to_csv(os.path.join(self.celeba_dir, bin_filename), index=False)

        print(f"Short content of file {bin_filename}", bin_dataframe.head(5), sep="\n", end="\n\n")
        print("Number of images:", bin_dataframe.shape[0])
        print("Categories:", bin_dataframe.columns)
        print("First attributes:", bin_dataframe.iloc[0])

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
