import torch.nn as nn
import torch
import math

from base import BaseModel
from utils.bbox import Toolbox
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
import pretrainedmodels as pm
import torch.optim as optim
import numpy as np
import utils.common_str as common_str


class FOTSModel:

    def __init__(self, config, is_train=False):
        self.mode = config['model']['mode']
        assert self.mode.lower() in ['recognition', 'detection', 'united'], f'模式[{self.mode}]不支持'
        keys = getattr(common_str, config['model']['keys'])
        if is_train:
            print('\nChoose to use ImageNet pretrained Network.\n')
            backbone_network = pm.__dict__['resnet50'](pretrained='imagenet')  # resnet50 in paper
            for param in backbone_network.parameters():
                param.requires_grad = config['need_grad_backbone']
        else:
            print('\nTrain from scratch.\n')
            backbone_network = pm.__dict__['resnet50'](pretrained=None)

        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0  # replace all nan/inf in gradients to zero

        if not self.mode == 'detection': # 不是检测模式，即识别模式或联合模式
            self.conv_rec = shared_conv.SharedConv(backbone_network, config)
            self.nclass = len(keys) + 1
            self.recognizer = Recognizer(self.nclass, config)
            self.recognizer.register_backward_hook(backward_hook)

        if not self.mode == 'recognition': # 不是识别模式，即检测模式或联合模式
            # 如果fine-tune使用在识别上预训练的模型，则检测backbone无需加入训练，直接用识别模块的即可
            self.conv_det = shared_conv.SharedConv(backbone_network, config) 
            self.detector = Detector(config)
            self.detector.register_backward_hook(backward_hook)

        self.roirotate = ROIRotate(config['model']['crnn']['img_h'])

    def available_models(self):
        '''
        Select different parts of the FOTSModel when using different mode
        '''
        if self.mode == 'detection':
            return ['conv_det', 'detector']
        elif self.mode == 'recognition':
            return ['conv_rec', 'recognizer']
        else:
            return ['conv_det', 'detector', 'conv_rec', 'recognizer'] # 双分支，两个backbone是因为共享backbone难以收敛

    def parallelize(self):
        for m_model in self.available_models():
            setattr(self, m_model, torch.nn.DataParallel(getattr(self, m_model)))

    def to(self, device):
        for m_model in self.available_models():
            setattr(self, m_model, getattr(self, m_model).to(device))

    def summary(self):
        for m_model in self.available_models():
            getattr(self, m_model).summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [{'params': getattr(self, m_model).parameters()} for m_model in self.available_models()],
            **params
        )
        return optimizer

    def train(self):
        for m_model in self.available_models():
            getattr(self, m_model).train()

    def eval(self):
        for m_model in self.available_models():
            getattr(self, m_model).eval()

    def state_dict(self):
        return {f'{m_ind}': getattr(self, m_model).state_dict()
                for m_ind, m_model in enumerate(self.available_models())}

    def load_state_dict(self, sd): # state_dict格式是一个dict: {'0' : part1, '1' : part2, ...}
        for m_ind, m_model in enumerate(self.available_models()):
            getattr(self, m_model).load_state_dict(sd[f'{m_ind}'])

    @property
    def training(self):
        return all([getattr(self, m_model).training for m_model in self.available_models()])

    def parameters(self):
        for m_module in [getattr(self, m_module) for m_module in self.available_models()]:
            for m_para in m_module.parameters():
                yield m_para

    def forward(self, image, boxes=None, mapping=None):
        """
        :param image:   图像
        :param boxes:   训练的时候gt的boxes
        :param mapping: 训练的时候boxes与图像的映射
        """
        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        def _compute_boxes(_score_map, _geo_map): # 从得到的两个map中恢复出原图的检测框，测试时才使用，训练不用
            score = _score_map.permute(0, 2, 3, 1)
            geometry = _geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy() # detach后放到cpu会导致张量计算图断开，无法反向传播
            geometry = geometry.detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            _pred_mapping = [] # 下标为i的样本有几个检测框，用于做_pred_boxes的索引
            _pred_boxes = [] # 所有预测出来的检测框
            for i in range(score.shape[0]):
                cur_score = score[i, :, :, 0]
                cur_geometry = geometry[i, :, :, ]
                detected_boxes, _ = Toolbox.detect(score_map=cur_score, geo_map=cur_geometry, timer=timer)
                if detected_boxes is None:
                    continue
                num_detected_boxes = detected_boxes.shape[0]

                if len(detected_boxes) > 0:
                    _pred_mapping.append(np.array([i] * num_detected_boxes))
                    _pred_boxes.append(detected_boxes)
            return np.concatenate(_pred_boxes) if len(_pred_boxes) > 0 else [], \
                   np.concatenate(_pred_mapping) if len(_pred_mapping) > 0 else []

        score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices = \
            None, None, (None, torch.Tensor(0)), boxes, mapping, mapping # 将各种数据都初始化

        # 三种模式前向传播
        if self.mode == 'detection':
            #print('Detection mode')
            feature_map_det = self.conv_det.forward(image)
            score_map, geo_map = self.detector(feature_map_det)
            if not self.training: # 测试模式使用预测出来的geo_map来推测bbox
                pred_boxes, pred_mapping = _compute_boxes(score_map, geo_map)
            else: # 训练模式不推测bbox，直接返回geo_map
                pred_boxes, pred_mapping = boxes, mapping

        elif self.mode == 'recognition':
            #print('Recognition mode')
            pred_boxes, pred_mapping = boxes, mapping # 识别模式直接将box告诉网络
            feature_map_rec = self.conv_rec.forward(image)
            rois, lengths, indices = self.roirotate(feature_map_rec, pred_boxes[:, :8], pred_mapping)
            preds = self.recognizer(rois, lengths).permute(1, 0, 2)
            lengths = torch.tensor(lengths).to(device)
            
        elif self.mode == 'united':
            #print('United mode')
            feature_map_det = self.conv_det.forward(image) # 如果直接训练联合模型，则需要双分支backbone
            #feature_map_det = self.conv_rec.forward(image) # 如果先在识别任务上预训练，那么直接拿识别backbone进行检测
            score_map, geo_map = self.detector(feature_map_det)
            if self.training: # 训练模式不推测bbox，直接将框的位置告诉识别模块
                pred_boxes, pred_mapping = boxes, mapping
            else: # 测试模式使用预测出来的geo_map来推测bbox
                pred_boxes, pred_mapping = _compute_boxes(score_map, geo_map)
            if len(pred_boxes) > 0: # 如果有要预测的文本，才执行识别模块
                #print('Detected {} boxes'.format(len(pred_boxes)))
                feature_map_rec = self.conv_rec.forward(image)
                rois, lengths, indices = self.roirotate(feature_map_rec, pred_boxes[:, :8], pred_mapping)
                preds = self.recognizer(rois, lengths).permute(1, 0, 2) # 识别标注好并完成roi rotate的框
                lengths = torch.tensor(lengths).to(device)
            else: # 如果当前图像没有文本要识别，则返回空预测
                #print('Didn\'t detect boxes')
                preds = torch.empty(1, image.shape[0], self.nclass, dtype=torch.float)
                lengths = torch.ones(image.shape[0])

        return score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices


class Recognizer(BaseModel):
    '''
    Recognize text with CRNN
    '''
    def __init__(self, nclass, config):
        super().__init__(config)
        crnn_config = config['model']['crnn']
        self.crnn = CRNN(crnn_config['img_h'], 32, nclass, crnn_config['hidden'])

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):
    '''
    Detect text and use geometry to represent rbox
    '''
    def __init__(self, config):
        super().__init__(config)
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feature_map):
        final = feature_map
        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0-1 的值是到上下左右的距离，但是图像他都缩放到512 * 512了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(geoMap) * 512

        # 这里将每像素对应的角度map单独求出
        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2 # 将angle map中的实数域数值变为-pi/4 ~ pi/4

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry # N*1*H*W, N*5*H*W
