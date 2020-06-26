# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import service.base_pb2 as base__pb2


class OCRStub(object):
  """定义OCR服务
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.detect_and_recognize = channel.unary_unary(
        '/OCR/detect_and_recognize',
        request_serializer=base__pb2.ImageRequest.SerializeToString,
        response_deserializer=base__pb2.OCRResponse.FromString,
        )
    self.detect = channel.unary_unary(
        '/OCR/detect',
        request_serializer=base__pb2.ImageRequest.SerializeToString,
        response_deserializer=base__pb2.OCRResponse.FromString,
        )
    self.recognize = channel.unary_unary(
        '/OCR/recognize',
        request_serializer=base__pb2.ImageRequest.SerializeToString,
        response_deserializer=base__pb2.OCRResponse.FromString,
        )


class OCRServicer(object):
  """定义OCR服务
  """

  def detect_and_recognize(self, request, context):
    """请求OCR识别
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def detect(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def recognize(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_OCRServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'detect_and_recognize': grpc.unary_unary_rpc_method_handler(
          servicer.detect_and_recognize,
          request_deserializer=base__pb2.ImageRequest.FromString,
          response_serializer=base__pb2.OCRResponse.SerializeToString,
      ),
      'detect': grpc.unary_unary_rpc_method_handler(
          servicer.detect,
          request_deserializer=base__pb2.ImageRequest.FromString,
          response_serializer=base__pb2.OCRResponse.SerializeToString,
      ),
      'recognize': grpc.unary_unary_rpc_method_handler(
          servicer.recognize,
          request_deserializer=base__pb2.ImageRequest.FromString,
          response_serializer=base__pb2.OCRResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'OCR', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
