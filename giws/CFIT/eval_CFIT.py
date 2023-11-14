import os
import pdb
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import utils
from model.model import TwitterClassifier
from model.dataset import GTDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, processer = utils.get_clip(device=device)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_or_path', type=str)
    parser.add_argument('--test_data_folder', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--result_file', type=str, default='./evaluation.json')
    # parser.add_argument('--device', type=int, help='the rank of GPU supposed to be used')
    # parser.add_argument('')
    args = parser.parse_args()
    return args

def load_model(model_path):
    model_data = torch.load(model_path)
    model = TwitterClassifier()
    ####################
    # lora TBD
    ####################
    model.load_state_dict(model_data['model'])
    return model.to(device)

def setup_dataset(test_data_folder, test_batch_size, processer):
    img_folder, text_folder, label_folder = tuple([os.path.join(test_data_folder, i) for i in ['img', 'text', 'label']])
    dataset = GTDataset(img_folder, text_folder, label_folder, processer)
    dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return dataloader

def eval(model_or_path, test_data_folder, test_batch_size, result_file=None, dataloader=None, is_return=False):
    if isinstance(model_or_path, nn.Module):
        model = model_or_path.to(device)
    elif os.path.exists(model_or_path):
        model = load_model(model_or_path)
    else:
        raise Exception('model error')
    if dataloader is None:
        dataloader = setup_dataset(test_data_folder, test_batch_size, processer)
    all_batch_length = len(dataloader)
    accs = []
    predictions = []
    all_labels = []

    def accuracy(prds, label):
        prds_ids = torch.argmax(prds, dim=1)
        predictions.extend(list(map(lambda x: x.item(), prds_ids)))
        label_ids = torch.argmax(label, dim=1)
        all_labels.extend(list(map(lambda x: x.item(), label_ids)))
        eq_num = (prds_ids == label_ids).sum().item() 
        return eq_num / prds_ids.numel()

    def confusion_matrix(y_true, y_pred, num_classes=None):
        if num_classes is None:
            num_classes = max(max(y_true), max(y_pred)) + 1

        # 初始化混淆矩阵
        conf_matrix = [[0] * num_classes for _ in range(num_classes)]

        # 填充混淆矩阵
        for true_label, pred_label in zip(y_true, y_pred):
            conf_matrix[true_label][pred_label] += 1

        return conf_matrix

    for it, (image, text, label) in enumerate(dataloader):
        output = model(image.to(device), text.to(device))
        acc = accuracy(output, label.to(device))
        print(f'Batch [{it+1}/{all_batch_length}] Accuracy: {round(acc, 4)}')
        accs.append(acc)

    ave_accuracy = sum(accs) / len(accs)
    confusion_m = confusion_matrix(all_labels, predictions)

    if is_return:
        return ave_accuracy, confusion_m

    print('---------------------------------------------')
    print(f'Eval finished. Average Accuracy: {ave_accuracy}')
    print('---------------------------------------------')  
    if result_file is not None:
        with open(args.result_file, 'w') as f:
            data = {
                'Accuracy': ave_accuracy,
                'batch_acc': accs,
                'predictions': predictions
            }
            json.dump(data, f)


def main(args):
    eval(**vars(args))


if __name__=='__main__':
    args = parse()
    main(args)
