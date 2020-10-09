from classifier.DenseNet import DenseNet121
from classifier.DataModule import DataModule
import os
import argparse
import yaml
import torch
import numpy as np
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/celebA_YSBBB_Classifier.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    classifier_output_path = os.path.join(config['log_dir'], config['name'], 'classifier_output')
    if not os.path.exists(classifier_output_path):
        os.mkdir(classifier_output_path)
    
    data_module = DataModule(config)
    
    train_csv = data_module.train_dataset.data
    test_csv = data_module.val_dataset.data
    
    train_loader = data_module.train_dataloader()
    test_loader = data_module.val_dataloader()

    model = DenseNet121(config)
    cls_ckpt_path = os.path.join('checkpoints/classifier', config['name'], 'last.ckpt')
    model.load_state_dict(torch.load(cls_ckpt_path)['state_dict'])
    
    device = 'cpu'
    if torch.cuda.is_available():
        print('Training on GPU...')
        device = 'cuda'
    
    model.to(device)
    
    model.eval()

    names = np.empty([0])
    prediction_y = np.empty([0])
    true_y = np.empty([0])

    inner = tqdm.tqdm(total=len(train_csv), desc='Train samples', position=0)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(train_loader):
            start = batch_idx * model.batch_size
            images, targets = images.to(device), targets.to(device)
            
            ns = train_csv.iloc[start: start + model.batch_size]['Path'].to_list()
        
            for i in range(len(ns)):
                ns[i] = os.path.join(data_module.image_dir, ns[i])
            
            predictions = torch.sigmoid(model(images))

            if batch_idx == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(predictions.cpu())
                true_y = np.asarray(targets.cpu())
            else:
                names = np.append(names, np.asarray(ns), axis=0)
                prediction_y = np.append(prediction_y, np.asarray(predictions.cpu()), axis=0)
                true_y = np.append(true_y, np.asarray(targets.cpu()), axis=0)

            inner.update(model.batch_size)

    np.save(classifier_output_path + '/name_train.npy', names)
    np.save(classifier_output_path + '/prediction_y_train.npy', prediction_y)
    np.save(classifier_output_path + '/true_y_train.npy', true_y)

    names = np.empty([0])
    prediction_y = np.empty([0])
    true_y = np.empty([0])

    inner = tqdm.tqdm(total=len(test_csv), desc='Test samples', position=1)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            start = batch_idx * model.batch_size
            images, targets = images.to(device), targets.to(device)
            
            ns = test_csv.iloc[start: start + model.batch_size]['Path'].to_list()
        
            for i in range(len(ns)):
                ns[i] = os.path.join(data_module.image_dir, ns[i])
            
            predictions = torch.sigmoid(model(images))

            if batch_idx == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(predictions.cpu())
                true_y = np.asarray(targets.cpu())
            else:
                names = np.append(names, np.asarray(ns), axis=0)
                prediction_y = np.append(prediction_y, np.asarray(predictions.cpu()), axis=0)
                true_y = np.append(true_y, np.asarray(targets.cpu()), axis=0)

            inner.update(model.batch_size)

        np.save(classifier_output_path + '/name_test.npy', names)
        np.save(classifier_output_path + '/prediction_y_test.npy', prediction_y)
        np.save(classifier_output_path + '/true_y_test.npy', true_y)


if __name__ == '__main__':
    main()
