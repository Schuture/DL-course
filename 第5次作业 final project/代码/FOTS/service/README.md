# GRPC服务端搭建

## 简介

目前grpc部分改动比较频繁，而且稳定性暂时未验证，不建议使用。

## HOW TO RUN
`python serve.py`

配置文件在代码中修改。默认配置文件为`config_deploy.json`。

## 接口定义
|函数名|入参|输出|示例|异常说明|
|--|--|--|--|--|
|detect_and_recognize|1. 图像的base64|JSON字符串|```{"mode": "detect_and_recognize", "code": 200, "result": "110"}```||
## 文件说明

```reStructuredText
|-- example_client # client例子
|   `-- client.py # python版client例子
|-- protos # proto接口文件夹
|   `-- base.proto # 基本的ocr服务接口
|-- base_pb2_grpc.py # 根据base.proto生成
|-- base_pb2.py # 根据base.proto生成
|-- __init__.py 
|-- README.md # 本文件
`-- server.py # 服务端实现
```

利用以下脚本用于生成pb2_grpc.py和pb2.py:

```bash
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/base.proto
```



### NOTE

在生成`_pb2_grpc.py`之后需要手动把`import base_pb2 as base__pb2` 改为`import service.base_pb2 as base__pb2`

