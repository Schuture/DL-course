import torch.nn as nn
import torch


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, lengths):
        self.rnn.flatten_parameters()
        total_length = input.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        recurrent, _ = self.rnn(packed_input)  # [T, b, h * 2]
        padded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(recurrent, total_length=total_length, batch_first=True)

        b, T, h = padded_input.size()
        t_rec = padded_input.contiguous().view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)
        output = nn.functional.log_softmax(output, dim=-1)  # required by pytorch's ctcloss

        return output


class HeightMaxPool(nn.Module):

    def __init__(self, size=(2, 1), stride=(2, 1)):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=size, stride=stride)

    def forward(self, input):
        return self.pooling(input)


class CRNN(nn.Module):

    def __init__(self, img_h, num_channel, num_class, num_hidden, leakyRelu=False):
        """
        初始化CRNN
        :param img_h:    图像高度，必须是8的倍数
        :param num_channel:  传入图像的channel的数量
        :param num_class:  字符数量+1
        :param num_hidden:  LSTM隐层的大小
        :param leakyRelu:   是否使用leakyRELU
        """
        super(CRNN, self).__init__()
        assert img_h % 8 == 0 and img_h > 0, '图像高度必须为8的正整数倍'
        self.col_size = img_h // 8
        kernel_sizes = [3, 3, 3, 3, 3, 3]
        pads = [1, 1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1, 1]
        kernel_nums = [72] * 6

        cnn = nn.Sequential()

        def convRelu(i):
            nIn = num_channel if i == 0 else kernel_nums[i - 1]
            nOut = kernel_nums[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, kernel_sizes[i], strides[i], pads[i]))

            cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        cnn.add_module('HeightMaxPooling{0}'.format(0), HeightMaxPool())
        convRelu(2)
        convRelu(3)
        cnn.add_module('HeightMaxPooling{0}'.format(1), HeightMaxPool())
        convRelu(4)
        convRelu(5)
        cnn.add_module('HeightMaxPooling{0}'.format(2), HeightMaxPool())

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(kernel_nums[-1]*self.col_size, num_hidden, num_class)

    def forward(self, input, lengths):
        # conv features B*C*H*W
        conv = self.cnn(input)
        _b, _c, _h, _w = conv.shape
        assert _h == self.col_size, '卷积结果与预期不符'

        # b, c, h, w_after = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # _, _, _, w_before = input.size()
        # step = (w_before / w_after).ceil()
        # padded_width_after = (lengths - 1 / step).ceil()

        # conv = conv.squeeze(2)
        conv = conv.reshape((-1, _c*_h,_w))
        conv = conv.permute(0, 2, 1)  # [B, T, C]

        # rnn features
        output = self.rnn(conv, lengths)

        return output
