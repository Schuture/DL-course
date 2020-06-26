import os
import math
import json
import logging
import torch
import torch.optim as optim
from utils.util import ensure_dir


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, finetune, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        
        # 设置设备
        if torch.cuda.is_available():
            if config['cuda']:
                self.with_cuda = True
                self.gpus = {i: item for i, item in enumerate(self.config['gpus'])}
                device = 'cuda'
                if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
                    self.model = torch.nn.DataParallel(self.model)
                torch.cuda.empty_cache()
            else:
                self.with_cuda = False
                device = 'cpu'
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)
        self.model.to(self.device).half()

        self.logger.debug('Model is initialized.')
        self._log_memory_usage() # 将当前显存占用加入到log中

        self.train_logger = train_logger
        
        # 由config中的字符来指定使用nn.optim中的哪个优化器
        self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(), **config['optimizer'])
        
        # 由config中的字符来指定使用nn.optim.lr_scheduler中的哪个学习率管理器
        self.lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_freq = config['lr_scheduler_freq']
            
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
                  
        if resume: # resume就载入整个ckpt，继续训练
            self._resume_checkpoint(resume)

        if finetune and not resume: # finetune就不载入ckpt中的模型参数外的数据
            self._restore_checkpoint(finetune)


    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                result = self._train_epoch(epoch) # 训练一个epoch，返回log
            except torch.cuda.CudaError:
                self._log_memory_usage()

            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics): # precision, recall, hmean
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics): # val_precision, val_recall, val_hmean
                        log['val_' + metric.__name__] = result['val_metrics'][i]
                else:
                    log[key] = value # loss...
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))
            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best)\
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch, log, save_best=True) # 如果当前epoch得到最好结果，则保存为model_best.pth.tar
            if epoch % self.save_freq == 0: # 每间隔一定epoch保存一次模型参数
                self._save_checkpoint(epoch, log)
            if self.lr_scheduler and epoch % self.lr_scheduler_freq == 0: # 每间隔一定epoch更新一次lr_scheduler状态
                self.lr_scheduler.step(epoch) # 学习率管理器将学习率调整到epoch处的学习率
                lr = self.lr_scheduler.get_lr()[0]
                self.logger.info('New Learning Rate: {:.8f}'.format(lr)) # 显示当前学习率

    def _log_memory_usage(self):
        '''
        Record the cuda memory usage to log
        '''
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                .format(epoch, log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # 将参数全部放入GPU
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.device)
        self.train_logger = checkpoint['logger']
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def _restore_checkpoint(self, checkpoint_path):
        """
        just load parameter of pretrained model

        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        pretrained_dict = checkpoint['state_dict'] # 预训练模型的state_dict
        model_dict = self.model.state_dict() # 当前用来训练的模型的state_dict
        
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
            print('Parameters are consistant, load state dict directly ...')
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # if self.with_cuda:
            #     for state in self.optimizer.state.values():
            #         for k, v in state.items():
            #             if isinstance(v, torch.Tensor):
            #                 state[k] = v.cuda(self.device)
        

def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict: 预训练模型的state_dict
    :param model_dict: 当前模型的state_dict
    :return: 预训练模型中包含的当前模型的state_dict
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else: # 预训练模型多出来的，但是本模型没有
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

