import os
import pdb
import json
import argparse

import torch
from torch.utils.data import DataLoader

import utils
from model.model import TwitterClassifier
from model.dataset import GTDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, processer = utils.get_clip(device)


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
    model.load_state_dict(model_data['model'])
    return model.to(device)

def setup_dataset(test_data_folder, test_batch_size, processer):
    img_folder, text_folder, label_folder = tuple([os.path.join(test_data_folder, i) for i in ['img', 'text', 'label']])
    dataset = GTDataset(img_folder, text_folder, label_folder, processer)
    dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return dataloader

def eval(model_or_path, test_data_folder, test_batch_size, result_file=None, dataloader=None, is_return=False):
    if isinstance(model_or_path, TwitterClassifier):
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

    def accuracy(prds, label):
        prds_ids = torch.argmax(prds, dim=1)
        predictions.extend(list(prds_ids))
        label_ids = torch.argmax(label, dim=1)
        eq_num = (prds_ids == label_ids).sum().item() 
        return eq_num / prds_ids.numel()

    for it, (image, text, label) in enumerate(dataloader):
        image_features, text_features = utils.get_features(clip_model, image.to(device), text.to(device))
        input_ids = torch.hstack((image_features, text_features)).to(torch.float32)
        output = model(input_ids)
        acc = accuracy(output, label.to(device))
        print(f'Batch [{it+1}/{all_batch_length}] Accuracy: {round(acc, 4)}')
        accs.append(acc)

    ave_accuracy = sum(accs) / len(accs)

    if is_return:
        return ave_accuracy

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
