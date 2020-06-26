import sys,os
sys.path.insert(0,os.path.abspath('..'))
import base64
import logging
import time
from concurrent import futures
from io import BytesIO

import grpc
import torch
from PIL import Image

import service.base_pb2_grpc as base_pb2_grpc
import service.base_pb2 as base_pb2
import json

from model.model import FOTSModel
from utils import common_str
from utils.bbox import Toolbox
from utils.util import strLabelConverter


class OCRServer(base_pb2_grpc.OCRServicer):

    def __init__(self, config):
        """
        负责服务初始化
        """
        self.model = FOTSModel(config, False)
        self.model.eval()
        self.config = config
        self.model.load_state_dict(torch.load(config['model_path'])['state_dict'])
        self.label_converter = strLabelConverter(getattr(common_str, self.config['model']['keys']))
        if config['cuda']:
            self.model.to(torch.device("cuda:0"))
        print('init finish')

    def detect(self, request, context):
        to_return = {'mode': 'detect'}
        return base_pb2.OCRResponse(message=json.dumps(to_return))

    def recognize(self, request, context):
        to_return = {'mode': 'recognize'}
        return base_pb2.OCRResponse(message=json.dumps(to_return))

    def _area_by_shoelace(self, points):
        x, y = [_[0] for _ in points], [_[1] for _ in points]
        return abs(sum(i * j for i, j in zip(x, y[1:] + y[:1]))
                   - sum(i * j for i, j in zip(x[1:] + x[:1], y))) / 2

    def detect_and_recognize(self, request, context):
        to_return = {'mode': 'detect_and_recognize'}
        to_process_img = Image.open(BytesIO(base64.b64decode(request.image))).convert('RGB')
        polys_and_texts, _, _ = Toolbox.predict(
            to_predict_img=to_process_img,
            model=self.model,
            with_img=False,
            output_dir=None,
            with_gpu=self.config['cuda'],
            output_txt_dir=None,
            labels=None,
            label_converter=self.label_converter)
        if polys_and_texts is not None and len(polys_and_texts) > 0:
            to_return['code'] = 200
            to_return['result'] = max(polys_and_texts, key=lambda x: self._area_by_shoelace(x[0]))[1]
        else:
            to_return['code'] = 201
            to_return['result'] = '未识别出'
        return base_pb2.OCRResponse(message=json.dumps(to_return))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 5 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 5 * 1024 * 1024)])
    with open('config_deploy.json', encoding='utf-8', mode='r') as to_read:
        server_conf = json.loads(to_read.read())
    base_pb2_grpc.add_OCRServicer_to_server(OCRServer(server_conf), server)
    server.add_insecure_port('[::]:%d' % server_conf['port'])
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
