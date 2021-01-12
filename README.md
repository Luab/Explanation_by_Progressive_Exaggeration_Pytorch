# Explanation by Progressive Exaggeration Pytorch
Unofficial PyTorch reimplementation of ICLR 2020 paper: *Explanation By Progressive Exaggeration*.

[**Paper**](https://openreview.net/forum?id=H1xFWgrFPS)

## Installation
```bash
$ pip install -r requirements.txt
```
## Usage
1. Prepare the dataset for training
```
./notebooks/PreprocessData.ipynb
```

2. Train a classifier. Skip this step if you have a pretrained classifier. <br/>
Training logs of the classifier are saved at: *./$log_dir$/$name$*. <br/>
Model checkpoints of the classifier are saved at: *./checkpoints/classifier/$name$* ($log_dir$ and $name$ are defined in the corresponding config file). <br/>

2.a. To train a multi-label classifier on all 40 attributes
```
python train_classifier.py --config 'configs/celebA_DenseNet_Classifier.yaml'
```
2.b. To train a binary classifier on 1 attribute
```
python train_classifier.py --config 'configs/celebA_Young_Classifier.yaml'
```

3. Process the output of the classifier and create input for Explanation model by discretizing the posterior probability. 
   The input data for the Explanation model is saved at: $log_dir$/$name$/explainer_input/
```
./notebooks/ProcessClassifierOutput.ipynb
```

4. Train explainer model. The output is saved at: $log_dir$/$name.
```
python train_explainer.py --config 'configs/celebA_Young_Explainer.yaml'
```

5. Explore the trained Explanation model and see qualitative results.
```
./notebooks/TestExplainer.ipynb
```

6. Save results of the trained Explanation model for quantitative experiments.
```
python test_explainer.py --config 'configs/celebA_Young_Explainer.yaml'
```

7. Use the saved results to perform experiments as shown in paper
```
./notebooks/Experiment_CelebA.ipynb 
```