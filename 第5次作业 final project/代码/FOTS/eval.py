#!/usr/local/bin/python
# -*-coding=utf-8 -*-
import argparse
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from model.model import FOTSModel
from utils import common_str
from utils.bbox import Toolbox
from utils.util import strLabelConverter

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu): # 载入模型、模型参数
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoint = torch.load(model_path)
    if not checkpoint:
        raise RuntimeError('No checkpoint found.')
    config = checkpoint['config']
    
    model = FOTSModel(config)
    
    pretrained_dict = checkpoint['state_dict'] # 预训练模型的state_dict
    model_dict = model.state_dict() # 当前用来训练的模型的state_dict
    
    if pretrained_dict.keys() != model_dict.keys(): # 需要进行参数的适配
        print('Parameters are inconsistant, adapting model parameters ...')
        # 在合并前(update),需要去除pretrained_dict一些不需要的参数
        # 只含有识别分支的预训练模型参数字典中键'0', '1'对应全模型参数字典中键'2', '3'
        pretrained_dict['2'] = transfer_state_dict(pretrained_dict['0'], model_dict['2'])
        pretrained_dict['3'] = transfer_state_dict(pretrained_dict['1'], model_dict['3'])
        del pretrained_dict['0'] # 把原本预训练模型中的键值对删掉，以免错误地更新当前模型中的键值对
        del pretrained_dict['1']
        model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
        self.model.load_state_dict(model_dict)
    else:
        print('Parameters are consistant, load state dict directly ...\n')
        model.load_state_dict(pretrained_dict)

    if with_gpu:
        model.to(torch.device("cuda:0"))
        model.parallelize()

    model.eval()
    return model


def load_annotation(gt_path):
    with gt_path.open(mode='r') as f:
        label = dict()
        label["coor"] = list() # 坐标
        label["ignore"] = list() # 是否忽略这个rbox中的文本
        for line in f:
            text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
            if text[9] == "###" or text[9] == "*": # 与ICDAR-2015不同的是，ICDAR-2019第十个元素才是文本，第九个是语言
                label["ignore"].append(True)
            else:
                label["ignore"].append(False)
            bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            label["coor"].append(bbox)
    return label


def main(args: argparse.Namespace):
    model_path = args.model
    image_dir = args.image_dir
    output_img_dir = args.output_img_dir
    output_txt_dir = args.output_txt_dir

    if output_img_dir is not None and not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if output_txt_dir is not None and not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    annotation_dir = args.annotation_dir
    with_image = True if output_img_dir else False # 是否输出预测的图片
    with_gpu = True if torch.cuda.is_available() and not args.no_gpu else False # 是否使用gpu

    model = load_model(model_path, with_gpu)
    if annotation_dir is not None: # 有标注文件就计算预测的各项指标

        true_pos, true_neg, false_pos, false_neg = [0] * 4
        for image_fn in tqdm(image_dir.glob('*.jpg')):
            gt_path = annotation_dir / image_fn.with_suffix('.txt').name # 直接将.jpg的文件改成.txt就是对应的标注了
            labels = load_annotation(gt_path)
            # try:
            with torch.no_grad(): # 计算模型在数据集上每个样本的预测值并保存预测图像、文本
                polys, im, res = Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, labels,
                                                 output_txt_dir, strLabelConverter(getattr(common_str,args.keys)))
            true_pos += res[0]
            false_pos += res[1]
            false_neg += res[2]
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        print("TP: %d, FP: %d, FN: %d, precision: %f, recall: %f" % (true_pos, false_pos, false_neg, precision, recall))
    else: # 没有标注文件就仅仅输出预测图像并保存
        with torch.no_grad():
            for image_fn in tqdm(image_dir.glob('*.jpg')):
                Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, None, None,
                                strLabelConverter(getattr(common_str,args.keys)))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model',
                        default='./model_best.pth.tar',
                        type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_img_dir', type=pathlib.Path,
                        default='visualization/output_img',
                        help='output dir for drawn images')
    parser.add_argument('-t', '--output_txt_dir', type=pathlib.Path,
                        default='visualization/output_text',
                        help='output dir for text prediction')
    parser.add_argument('-i', '--image_dir', default='visualization/img',
                        type=pathlib.Path,
                        help='dir for input images')
    parser.add_argument('-a', '--annotation_dir', default='visualization/gt',
                        type=pathlib.Path,
                        help='dir for input annotation')
    parser.add_argument('-k', '--keys',
                        default='DL_str',
                        type=str,
                        help='keys in common_str')
    parser.add_argument('--no_gpu', # 如果显存不够大，那就只能用cpu
                        action='store_true', # 这个参数只是起开关的作用，无需指定具体值
                        help='whether to use gpu')

    args = parser.parse_args()
    main(args)
