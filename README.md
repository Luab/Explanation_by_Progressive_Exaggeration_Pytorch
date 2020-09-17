# Explanation_by_Progressive_Exaggeration_Pytorch
Unofficial PyTorch reimplementation of ICLR 2020 paper: *Explanation By Progressive Exaggeration*.

[**Paper**](https://openreview.net/forum?id=H1xFWgrFPS)

## Installation
```bash
$ pip install -r requirements.txt
```
## Usage
1. Download the CelebA dataset (*img_align_celeba.zip*) to the *./data* folder. <br/>
Prepare data for the training of classifier
```
python prepare_data_for_classifier.py
```
Images are placed to the *./data/CelebA/images* and *.txt* files with their names (paths) and labels are stored in the *./data/CelebA*. These text files are used as input data to train the classifier. <br/>

2. Train a classifier. Skip this step if you have a pretrained classifier. <br/>
Training logs of the classifier are saved at: *./$log_dir$/$name$*. <br/>
Model checkpoints of the classifier are saved at: *./checkpoints/classifier/$name$* ($log_dir$ and $name$ are defined in the corresponing config file). <br/>
For viewing the classifier's training logs it is needed to launch the TensorBoard from the *./$log_dir$* folder in this way:. <br/>
```
tensorboard --logdir $name$
```
2.a. To train a multi-label classifier on all 40 attributes
```
python train_classifier.py --config 'configs/celebA_DenseNet_Classifier.yaml'
```
2.b. To train a binary classifier on 1 attribute
```
python train_classifier.py --config 'configs/celebA_Smile_DenseNet_Classifier.yaml'
```
