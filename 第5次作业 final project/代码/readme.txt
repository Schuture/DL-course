CRNN：参考https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
EAST：参考https://github.com/SakuraRiven/EAST
FOTS：参考https://github.com/novioleo/FOTS（有bug）

calculate_metrics.py：使用EAST输出数据集的检测结果txt后，计算与gt框的IoU，然后评估准确率、召回率、F measure

combine_state_dict.py：分别训练了FOTS的两个分支以后，合并两个分支的state_dict到一个模型中，作为最终模型

extract_chars.py：将训练集中出现过的字符全部提取出来，作为文本识别的字符集

extract_loss_from_log.py：使用nohup训练模型，然后从nohup.out中提取出损失值，并画图

make_ocr_dataset.py：使用gt从数据集图片中进行文本区域的截图并转化为矩形存储起来，用于专门训练CRNN