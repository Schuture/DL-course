import torch.optim as optim
import time
from pathlib import Path
import os
import torch

def get_optimizer(config, model):

    optimizer = None
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    lr = config.TRAIN.LR
    if config.TRAIN.RESUME.IS_RESUME:
        parameters = [{'params': parameters, 'initial_lr': lr}]
    momentum = config.TRAIN.MOMENTUM
    weight_decay = config.TRAIN.WD
    nesterov = config.TRAIN.NESTEROV

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

def create_log_folder(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'

    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)


    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        string = list(d.labels[idx].values())[0]
        string = string if len(string) > 0  else "###"
        label.append(string)
    return label


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        
        self.num_alphabet = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                try:
                    index = self.dict[char]
                except Exception as e:
                    if char not in self.dict:
                        # self.dict[char] = self.num_alphabet + 1
                        # self.num_alphabet += 1
                        # index = self.dict[char]
                        index = self.dict['#']
                    print("{} not in alphabet -- Resolved".format(char))

                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

def get_char_dict(path):
    with open(path, 'rb') as file:
        char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}