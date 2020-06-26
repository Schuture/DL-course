### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn
from torch.nn import CTCLoss


class DetectionLoss(nn.Module):

    def __init__(self):
        super(DetectionLoss, self).__init__()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask):
        classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt) # 真实rbox面积
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred) # 预测rbox面积
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0)) # 负对数似然损失
        L_theta = 1 - torch.cos(theta_pred - theta_gt) # 真实角与预测角度的余弦损失
        L_g = L_AABB + 20 * L_theta # 原论文中lambda=10，这里取20使得角度收敛更快

        #return torch.mean(L_g * y_true_cls * training_mask) + classification_loss # mask用于OHEM样本平衡去除不关心的样本
        # 原代码是上面一行，有错误
        none_zero = torch.nonzero(y_true_cls * training_mask).shape[0] # 可能是正例的数量
        if none_zero == 0:
            return 10 * torch.mean(L_g * y_true_cls * training_mask) + classification_loss
        return torch.sum(L_g * y_true_cls * training_mask) / none_zero + classification_loss

    def __dice_coefficient(self, y_true_cls, y_pred_cls,
                           training_mask):
        '''
        dice loss for classification，这里分类为二分类：前景/背景
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
        union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss()  # pred, pred_len, labels, labels_len

    def forward(self, *input):
        gt, pred = input[0], input[1]
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['mode']
        self.detection_loss = DetectionLoss()
        self.recognition_loss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                y_true_recog, y_pred_recog,
                training_mask):

        recognition_loss = torch.tensor([0]).float()
        detection_loss = torch.tensor([0]).float()

        # 三种模式有不同的损失
        if self.mode == 'recognition':
            recognition_loss = self.recognition_loss(y_true_recog, y_pred_recog)
        elif self.mode == 'detection':
            detection_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                 y_true_geo, y_pred_geo, training_mask)
        elif self.mode == 'united':
            detection_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                y_true_geo, y_pred_geo, training_mask)
            if y_true_recog: # gt存在才计算识别损失，因为有模型识别出有文本的地方实际上是没有的
                recognition_loss = self.recognition_loss(y_true_recog, y_pred_recog)

        recognition_loss = recognition_loss.to(detection_loss.device) # 将两个损失放入同一个设备
        return detection_loss, recognition_loss
