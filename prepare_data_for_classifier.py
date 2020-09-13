import os
import zipfile

import pandas as pd
import yaml


# Think about
# output_dir = os.path.join(config['log_dir'], config['name'])

# Read data file = df.iloc: popping attrubutes by their name

def prepare_data(config_path='configs/celebA_YSBBB_Classifier.yaml'):
    data_dir = "./data"
    celeba_dir = os.path.join(data_dir, "CelebA")
    zip_path = os.path.join(data_dir, "img_align_celeba.zip")
    zip_save = image_dir = os.path.join(celeba_dir, "images")
    text_dir = os.path.join(celeba_dir, "list_attr_celeba.txt")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert os.path.isfile(zip_path), f"File img_align_celeba.zip does not lie in {celeba_dir} folder!"

    print("Config  file:", str(config), sep='\n')

    # Checking that path images is not empty
    if os.path.exists(zip_save):
        if len(os.listdir(zip_save)) == 202599:
            print("Everything is okay, data is already preprocessed!")
            return
        else:
            print(f"Something went wrong: found {len(os.listdir(zip_save))} files, "
                  f"deleting folder and trying again!")
            os.remove(zip_save)

    # Unpacking archive and moving it into './data/CelebA/images'
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(celeba_dir)



    os.rename(os.path.join(celeba_dir, "img_align_celeba"), zip_save)
    assert os.path.isdir(image_dir)

    print("Image dir:", image_dir)
    print("Text dir:", text_dir)

    data_df = pd.read_csv(text_dir)
    print(f"Short content of file {text_dir}", data_df.head(5), sep="\n", end="\n\n")

    print("Number of images:", data_df.shape[0])
    print("Categories:", data_df.columns)
    print("First attributes:", data_df.iloc[0])

    if not os.path.isfile(os.path.join(celeba_dir, "Young_binary_classification.txt")):
        # Write the label file for target attribute binary classification
        attributes = "Young"
        bin_dataframe = data_df[["Path", attributes]]
        bin_filename = attributes + "_binary_classification.txt"
        bin_dataframe.to_csv(os.path.join(celeba_dir, bin_filename), index=False)

        print(f"Short content of file {bin_filename}", bin_dataframe.head(5), sep="\n", end="\n\n")
        print("Number of images:", bin_dataframe.shape[0])
        print("Categories:", bin_dataframe.columns)
        print("First attributes:", bin_dataframe.iloc[0])

if __name__ == '__main__':
    prepare_data()