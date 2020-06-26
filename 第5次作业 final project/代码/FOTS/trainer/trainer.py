import numpy as np
import torch

from base import BaseTrainer
from utils.bbox import Toolbox
from utils.common_str import custom_1
from utils.util import strLabelConverter


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, metrics,
                 finetune, resume, config,
                 data_loader, toolbox: Toolbox, valid_data_loader=None, train_logger=None,
                 keys=custom_1):
        super(Trainer, self).__init__(model, loss, metrics, finetune, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.toolbox = toolbox
        self.labelConverter = strLabelConverter(keys)

    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device).half())
        return t

    def _eval_metrics(self, pred, gt):
        precious, recall, hmean = self.metrics[0](pred, gt)
        return np.array([precious, recall, hmean])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_det_loss = 0
        total_rec_loss = 0
        total_metrics = np.zeros(3)  # precious, recall, hmean
        dataset_size = len(self.data_loader)
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                image_paths, img, score_map, geo_map, training_mask, transcripts, boxes, mapping = gt
                img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)

                self.optimizer.zero_grad()
                pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model.forward(img,
                                                                                                                 boxes,
                                                                                                                 mapping)
                
                if indices is not None: # 在预测得到的一系列框中选择indices包含的那部分，以保证输入loss的gt, pred能对齐
                    indice_transcripts = transcripts[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_mapping = pred_mapping[indices]
                    labels, label_lengths = self.labelConverter.encode(indice_transcripts.tolist())
                    recog = (labels, label_lengths)
                else:
                    recog = (None,None)

                det_loss, reg_loss = self.loss(score_map,
                                               pred_score_map if pred_score_map is not None else score_map,
                                               geo_map,
                                               pred_geo_map if pred_geo_map is not None else geo_map,
                                               recog,
                                               pred_recog,
                                               training_mask)
                # 论文中这里的两个loss地位等同，lambda=1。可以尝试0.1，因为两个Loss的抖动大约3:1
                loss = det_loss # 只对检测分支进行反向传播
                #loss = det_loss + 0.1 * reg_loss 
                loss.backward()
                self.optimizer.step()

                # bad case定位
                if det_loss.item() >= 10.0:
                    print('bad case: ', image_paths)
                #if reg_loss.item() >= 10.0:
                #    print('bad case: ', image_paths)
                    
                verified_det_loss = det_loss.item() if det_loss.item() <= 10.0 else 10.0 # 可能会有不好的样本使得损失爆炸
                verified_rec_loss = reg_loss.item() if reg_loss.item() <= 10.0 else 10.0 # 可能会有不好的样本使得损失爆炸
                total_det_loss += verified_det_loss
                total_rec_loss += verified_rec_loss
                verified_loss = verified_det_loss + verified_rec_loss
                total_loss += verified_loss
                
                pred_transcripts = []
                pred_fns = [image_paths[i] for i in pred_mapping]

                # 将预测值解码为文本
                if len(pred_mapping) > 0 and pred_recog[0] is not None:
                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        cur_text_len = lengths[i]
                        cur_text_pred = pred[:cur_text_len, i]
                        t = self.labelConverter.decode(cur_text_pred, cur_text_len)
                        pred_transcripts.append(t)
                    pred_transcripts = np.array(pred_transcripts)

                # 计算precision, recall, hmean
                gt_fns = [image_paths[i] for i in mapping]
                total_metrics += self._eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                                                    (boxes, transcripts, gt_fns))

                # 记录log
                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Detection Loss: {:.6f} Recognition Loss:{:.6f}'.format(
                            epoch,
                            batch_idx * self.data_loader.batch_size,
                            len(self.data_loader) * self.data_loader.batch_size,
                            100.0 * batch_idx / len(self.data_loader),
                            verified_loss, verified_det_loss, verified_rec_loss))
            except:
                print(image_paths) # 训练如果不成功，就要在报错之前打印出报错的那张图片路径
                raise

        log = {
            'loss': total_loss / dataset_size,
            'det_loss': total_det_loss / dataset_size,
            'rec_loss': total_rec_loss / dataset_size,
            'precious': total_metrics[0] / dataset_size,
            'recall': total_metrics[1] / dataset_size,
            'hmean': total_metrics[2] / dataset_size
        }

        if self.valid:
            val_log = self._valid_epoch() # 验证
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_metrics = np.zeros(3)
        with torch.no_grad():
            total_det_loss = 0
            total_rec_loss = 0
            print('Start validate')
            dataset_size = len(self.valid_data_loader)

            for batch_idx, gt in enumerate(self.valid_data_loader):
                image_paths, img, score_map, geo_map, training_mask, transcripts, boxes, mapping = gt
                try:
                    img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)
                    pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model.forward(
                        img, boxes, mapping)
                    if indices is not None and len(indices) <= len(transcripts):
                        indice_transcripts = transcripts[indices]
                        labels, label_lengths = self.labelConverter.encode(indice_transcripts.flatten().tolist())
                        recog = (labels, label_lengths)

                        det_loss, reg_loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, recog, pred_recog,
                                                       training_mask)

                        total_det_loss += det_loss.item()
                        if reg_loss.item() == np.inf: # 正确类别概率预测为0时，负对数损失会有无穷大的情况
                            total_rec_loss += 100
                        else:
                            total_rec_loss += reg_loss.item()
                    else: # 没有预测出边界框，或者是边界框数量大于图中的文本数量了
                        total_det_loss += 100
                        total_rec_loss += 100
                    
                    # 计算得出预测的文本
                    pred_transcripts = []
                    pred_fns = []
                    if len(pred_mapping) > 0 and pred_recog[0] is not None:
                        pred_mapping = pred_mapping[indices]
                        pred_boxes = pred_boxes[indices]
                        pred_fns = [image_paths[i] for i in pred_mapping]

                        pred, lengths = pred_recog
                        _, pred = pred.max(2)
                        for i in range(lengths.numel()):
                            l = lengths[i]
                            p = pred[:l, i]
                            t = self.labelConverter.decode(p, l)
                            pred_transcripts.append(t)
                        pred_transcripts = np.array(pred_transcripts)

                    gt_fns = [image_paths[i] for i in mapping]
                    total_val_metrics += self._eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                                                            (boxes, transcripts, gt_fns))
                except Exception as e: # 验证时如果没有成功则不报错，因为验证不是必须的
                    print(image_paths, e)
                    # raise

        return { # 如果验证没有成功，则会返回全0的log
            'val_loss': (total_rec_loss + total_det_loss) / dataset_size,
            'val_det_loss': total_det_loss / dataset_size,
            'val_rec_loss': total_rec_loss / dataset_size,
            'val_precious': total_val_metrics[0] / dataset_size,
            'val_recall': total_val_metrics[1] / dataset_size,
            'val_hmean': total_val_metrics[2] / dataset_size
        }
