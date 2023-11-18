import os
import pdb
import json
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from giws.CFIT.model import utils
from giws.CFIT.model.model import TwitterClassifier
from giws.CFIT.model.dataset import GTDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, processer = utils.get_clip(device=device)


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

def eval(model_or_path, test_data_folder, test_batch_size, result_file=None, dataloader=None, is_return=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    right_data = []
    false_data = []

    def accuracy(prds, label):
        prds_ids = torch.argmax(prds, dim=1)
        predictions.extend(list(map(lambda x: x.item(), prds_ids)))
        label_ids = torch.argmax(label, dim=1)
        all_labels.extend(list(map(lambda x: x.item(), label_ids)))
        eq_num = (prds_ids == label_ids).sum().item() 
        right_ids = torch.nonzero(torch.eq(prds_ids, label_ids), as_tuple=False).squeeze()
        false_ids = torch.nonzero(~torch.eq(prds_ids, label_ids), as_tuple=False).squeeze()
        return eq_num / prds_ids.numel(), right_ids, false_ids

    def confusion_matrix(y_true, y_pred, num_classes=None):
        if num_classes is None:
            num_classes = max(max(y_true), max(y_pred)) + 1

        # 初始化混淆矩阵
        conf_matrix = [[0] * num_classes for _ in range(num_classes)]

        # 填充混淆矩阵
        for true_label, pred_label in zip(y_true, y_pred):
            conf_matrix[true_label][pred_label] += 1

        return conf_matrix
    
    def compute_matrix(y_true, y_pred, num_classes=3):
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true][pred] += 1

        precision = [0] * num_classes
        recall = [0] * num_classes
        f1_score = [0] * num_classes

        for i in range(num_classes):
            true_positive = confusion_matrix[i][i]
            false_positive = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
            false_negative = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

            precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
            recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

        macro_precision = round(sum(precision) / num_classes, 3) if num_classes != 0 else 0
        macro_recall = round(sum(recall) / num_classes, 3) if num_classes != 0 else 0
        macro_f1_score = round(sum(f1_score) / num_classes,3) if num_classes != 0 else 0

        return macro_precision, macro_recall, macro_f1_score


    for it, (image, text, label) in enumerate(dataloader):
        output = model(image.to(device), text.to(device))
        acc, right_index, false_index = accuracy(output, label.to(device))
        # logging.info(f'Batch [{it+1}/{all_batch_length}] Accuracy: {round(acc, 4)}')
        accs.append(acc)

    ave_accuracy = sum(accs) / len(accs)
    confusion_m = confusion_matrix(all_labels, predictions)
    precision, recall, f1_score = compute_matrix(all_labels, predictions)

    if is_return:
        return ave_accuracy, confusion_m, precision, recall, f1_score

    print('---------------------------------------------')
    print(f'Eval finished. Average Accuracy: {ave_accuracy}')
    print('---------------------------------------------')  
    if result_file is not None:
        with open(args.result_file, 'w') as f:
            data = {
                'Accuracy': ave_accuracy,
                'confusion_matrix': confusion_m,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            json.dump(data, f)


def main(args):
    eval(**vars(args))


if __name__=='__main__':
    args = parse()
    main(args)
