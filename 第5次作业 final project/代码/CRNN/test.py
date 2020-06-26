import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets

import yaml
from easydict import EasyDict as edict
import argparse
import sys


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='./lib/config/OWN_config.yaml')
    parser.add_argument('-ckpt', '--checkpoint', default="output/OWN/crnn/2020-06-12-06-56_train52/checkpoints/checkpoint_52_acc_0.2105.pth", type=str)
    parser.add_argument('--mode', default="test", type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # ratio resize
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    h, w = img.shape
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(config, data_loader, dataset, converter, model, criterion, device, mode="test"):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(data_loader):

            labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(0, i, len(data_loader)), end="\r")
    print()

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_tests = len(data_set)
    accuracy = n_correct / num_tests
    print('Loss: {:.4f}, ncorrect: {}, num_tests: {}, accuray: {:.4f}'.format(losses.avg, n_correct, num_tests, accuracy))

    return accuracy


if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model = crnn.get_crnn(config).to(device)
    # print('loading pretrained model from {0}'.format(args.checkpoint))
    # model.load_state_dict(torch.load(args.checkpoint))

    model_state_file = args.checkpoint
    if model_state_file == '':
        print(" => no checkpoint found")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # converter
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)  # get corpus

    # define loss function
    criterion = torch.nn.CTCLoss()

    if args.mode == "train":
        data_set = get_dataset(config)(config, is_train=True)
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    elif args.mode == "test":
        data_set = get_dataset(config)(config, is_train=False)
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=config.TEST.BATCH_SIZE_PER_GPU,
            shuffle=config.TEST.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
    else:
        print("Wrong mode, choose from [test, train]")
        sys.exit(0)


    started = time.time()

    # img = cv2.imread(args.image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    # recognition(config, img, model, converter, device)

    acc = validate(config, data_loader, data_set, converter, model, criterion, device, mode=args.mode)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

