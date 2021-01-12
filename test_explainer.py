from explainer.Explainer import Explainer
from classifier.DataModule import DataModule
import os
import argparse
import yaml
import torch
import numpy as np
import tqdm


def convert_ordinal_to_binary(y, n):
    y = y.int()
    new_y = torch.zeros([y.shape[-1], n])
    new_y[:, 0] = y
    for i in range(0, y.shape[0]):
        for j in range(1, y[i] + 1):
            new_y[i, j] = 1
    return new_y.float()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['batch_size'] = config['num_bins']
    
    count_to_save = config['count_to_save']
    n_bins = config['num_bins']
    assets_dir = os.path.join(config['log_dir'], config['name'])
    ckpt_dir = os.path.join('./checkpoints/explainer', config['name'], 'last.ckpt')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    data_module = DataModule(config, to_explainer=True)
    test_loader = data_module.val_dataloader()
    test_csv = data_module.val_dataset.df
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    explainer = Explainer(config)
    exp_ckpt_path = os.path.join('checkpoints/explainer', config['name'], 'last.ckpt')
    explainer.load_state_dict(torch.load(exp_ckpt_path)['state_dict'])
    explainer.eval()
    explainer = explainer.to(device)

    real_img = np.empty([0])
    fake_images = np.empty([0])
    embedding = np.empty([0])
    s_embedding = np.empty([0])
    recons = np.empty([0])
    real_pred = np.empty([0])
    fake_pred = np.empty([0])
    recons_pred = np.empty([0])
    names = np.empty([0]) 

    inner = tqdm.tqdm(total=count_to_save, desc='Test samples', position=0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx == 2:
                break
                
            start = batch_idx * explainer.batch_size
            ns = test_csv.iloc[start : start + explainer.batch_size]['Path'].to_list()
            
            images, labels = batch
            images_repeat = torch.repeat_interleave(images, n_bins, 0)
            
            labels = torch.reshape(labels, shape=[-1])
            labels = torch.repeat_interleave(labels, n_bins, 0)
            source_labels = convert_ordinal_to_binary(labels, n_bins)
            
            target_labels = torch.tensor([range(n_bins) for j in range(images.shape[0])])
            target_labels = torch.reshape(target_labels, shape=[-1]) 
            target_labels = convert_ordinal_to_binary(target_labels, n_bins)
            
            fake_target_img, fake_target_img_embedding = explainer.G(images_repeat.to(device), target_labels[:, 0].to(device))
            fake_source_img, fake_source_img_embedding = explainer.G(fake_target_img.to(device), source_labels[:, 0].to(device))
            real_img_cls_prediction = torch.sigmoid(explainer.model(images_repeat.to(device)))
            fake_img_cls_prediction = torch.sigmoid(explainer.model(fake_target_img.to(device)))
            real_img_recons_cls_prediction = torch.sigmoid(explainer.model(fake_source_img.to(device)))
            fake_source_recons_img, x_source_img_embedding = explainer.G(images_repeat.to(device), source_labels[:, 0].to(device))
            
            if batch_idx == 0:
                real_img = images.cpu()
                fake_images = fake_target_img.cpu()
                embedding = fake_target_img_embedding.cpu()
                s_embedding = x_source_img_embedding.cpu()
                recons = fake_source_img.cpu()
                real_pred = real_img_cls_prediction.cpu()
                fake_pred = fake_img_cls_prediction.cpu()
                recons_pred = real_img_recons_cls_prediction.cpu()
                names = np.asarray(ns)
            else:
                real_img = np.append(real_img, images.cpu(), axis=0)
                fake_images = np.append(fake_images, fake_target_img.cpu(), axis=0)
                embedding = np.append(embedding, fake_target_img_embedding.cpu(), axis=0)
                s_embedding = np.append(s_embedding, x_source_img_embedding.cpu(), axis=0)
                recons = np.append(recons, fake_source_img.cpu(), axis=0)
                real_pred = np.append(real_pred, real_img_cls_prediction.cpu(), axis=0)
                fake_pred = np.append(fake_pred, fake_img_cls_prediction.cpu(), axis=0)
                recons_pred = np.append(recons_pred, real_img_cls_prediction.cpu(), axis=0)
                names = np.append(names, np.asarray(ns), axis=0)

            inner.update(explainer.batch_size)

    np.save(os.path.join(test_dir + '/real_img.npy'), real_img)
    np.save(os.path.join(test_dir + '/fake_images.npy'), fake_images)
    np.save(os.path.join(test_dir + '/embedding.npy'), embedding)
    np.save(os.path.join(test_dir + '/s_embedding.npy'), s_embedding)
    np.save(os.path.join(test_dir + '/recons.npy'), recons)
    np.save(os.path.join(test_dir + '/names.npy'), names)
    np.save(os.path.join(test_dir + '/real_pred.npy'), real_pred)    
    np.save(os.path.join(test_dir + '/fake_pred.npy'), fake_pred)    
    np.save(os.path.join(test_dir + '/recons_pred.npy'), recons_pred) 

    
if __name__ == '__main__':
    main()
