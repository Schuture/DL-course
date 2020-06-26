from __future__ import print_function

import base64
import logging
from io import BytesIO

import grpc
from PIL import Image

import service.base_pb2 as base_pb2
import service.base_pb2_grpc as base_pb2_grpc


def run(image: Image.Image):
    with grpc.insecure_channel('localhost:10001', options=[
        ('grpc.max_send_message_length', 5 * 1024 * 1024),
            ('grpc.max_receive_message_length', 5 * 1024 * 1024)]) as channel:
        stub = base_pb2_grpc.OCRStub(channel)
        ret = BytesIO()
        image.save(ret, image.format)
        ret.seek(0)
        img_base_64 = base64.b64encode(ret.getvalue())
        response = stub.detect_and_recognize(base_pb2.ImageRequest(image=img_base_64))
    print(response.message)


if __name__ == '__main__':
    logging.basicConfig()
    img = Image.open('/data/OCR/自有数据/test/o_1cg8pa56u14211010023054330233370.jpg')
    run(img)
