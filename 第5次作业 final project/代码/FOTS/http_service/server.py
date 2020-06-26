import sys, os

sys.path.insert(0, os.path.abspath('..'))
import json
from argparse import ArgumentParser
from io import BytesIO

import torch
import tornado.httpserver
import tornado.ioloop
import tornado.web
import requests
from PIL import Image

from model.model import FOTSModel
from utils import common_str
from utils.bbox import Toolbox
from utils.util import strLabelConverter

global model, label_converter, with_gpu


def get_bound_box(polygon):
    assert len(polygon) == 8, 'polygon size error'
    x, y = polygon[::2], polygon[1::2]
    pad_x = min(x)
    pad_y = min(y)
    width = max(x) - pad_x
    height = max(y) - pad_y
    return [int(pad_x), int(pad_y), int(height), int(width)]


class DetectHandler(tornado.web.RequestHandler):

    def post(self):
        img_url = self.get_argument("img_url", default=None, strip=False)
        to_return = {}
        detected_boxes = []
        try:
            img_content = requests.get(img_url, timeout=5).content
            to_process_img = Image.open(BytesIO(img_content))
            detected_boxes, _, _ = Toolbox.predict(
                to_predict_img=to_process_img,
                model=model,
                with_img=False,
                output_dir=None,
                with_gpu=with_gpu,
                output_txt_dir=None,
                labels=None,
                label_converter=label_converter)
            to_return['code'] = 200
        except requests.exceptions.RequestException as re:
            to_return['code'] = -1
        except Exception as e:
            to_return['code'] = -2
        finally:
            to_return['detect_nums'] = len(detected_boxes)
            to_return['bounding_boxes'] = []
            to_return['boxes'] = []
            for m_box, _ in detected_boxes:
                to_return['bounding_boxes'].append(
                    {m_name: m_value for m_name, m_value in
                     zip(['left', 'top', 'height', 'width'], get_bound_box(m_box.flatten()))})
                to_return['boxes'].append({m_name: m_value.tolist() for m_name, m_value in
                                           zip(['left_top', 'right_top', 'right_bottom', 'left_bottom'], m_box)})
            self.write(json.dumps(to_return))


if __name__ == "__main__":
    ag = ArgumentParser()
    ag.add_argument("-c", type=str, help='path to config file')
    args = ag.parse_args()

    with open(args.c, mode='r', encoding='utf-8') as to_read:
        config = json.loads(to_read.read())

    model = FOTSModel(config, False)
    model.eval()
    model.load_state_dict(torch.load(config['model_path'])['state_dict'])
    label_converter = strLabelConverter(getattr(common_str, config['model']['keys']))
    with_gpu = config['cuda']
    if with_gpu:
        model.to(torch.device("cuda:0"))

    routes = [
        ('/detect', DetectHandler),
        ('/recognize', DetectHandler),
        ('/detect_and_recognize', DetectHandler),
    ]

    application = tornado.web.Application(routes)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(config['port'])
    print('init finish')
    tornado.ioloop.IOLoop.current().start()
