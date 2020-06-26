import argparse
import json
import logging
import os
import pathlib
import random
from data_loader import SynthTextDataLoaderFactory
from data_loader import OCRDataLoaderFactory
from data_loader.dataset import ICDAR, MyDataset
from logger import Logger
from model.loss import *
from model.model import *
from model.metric import *
from trainer import Trainer
from utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume):
    train_logger = Logger()

    if config['data_loader']['dataset'] == 'icdar2015':
        # ICDAR 2015
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        ICDARDataset2015 = ICDAR(data_root, year='2015')
        data_loader = OCRDataLoaderFactory(config, ICDARDataset2015)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'synth800k':
        data_loader = SynthTextDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'mydataset':
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        data_loader = OCRDataLoaderFactory(config, MyDataset(data_root))
        train = data_loader.train()
        val = data_loader.val()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = eval(config['arch'])(config, is_train=True) # 将模型的字符串转化为表达式调用，然后实例化
    #model.summary() # 模型结构较为复杂，可以选择不打印模型结构，注释掉即可

    loss = eval(config['loss'])(config['model'])
    metrics = [eval(metric) for metric in config['metrics']]

    finetune_model = config['finetune']

    trainer = Trainer(model, loss, metrics,
                      finetune=finetune_model, # 导入预训练的模型，在此基础上继续训练
                      resume=resume,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      train_logger=train_logger,
                      toolbox = Toolbox,
                      keys=getattr(common_str,config['model']['keys']))

    trainer.train()


if __name__ == '__main__':
    
    SEED = 1228
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='./config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    # 如果有resume参数则不仅导入之前保存的模型参数，所有模型设置也会导入
    # checkpoint文件包含一个dict，keys=['arch', 'epoch', 'logger', 'state_dict', 'optimizer', 'monitor_best', 'config']
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        #assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None
    
    print('\nConfig is in the following:')
    print(config)

    main(config, args.resume)
