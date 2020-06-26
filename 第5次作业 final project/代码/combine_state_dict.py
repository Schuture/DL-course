import torch

recognition_ckpt_path = "/root/CYX_Space/FOTS-master-failed/saved_model/CYX第二次识别续训练/checkpoint-epoch068-loss-inf.pth.tar"
detection_ckpt_path = "/root/CYX_Space/FOTS-master/saved_model/CYX训练第二次检测分支/model_best.pth.tar"

rec_ckpt = torch.load(recognition_ckpt_path)
det_ckpt = torch.load(detection_ckpt_path)
rec_state_dict = rec_ckpt['state_dict']
det_state_dict = det_ckpt['state_dict']

# ['conv_det', 'detector', 'conv_rec', 'recognizer']
state_dict = {'0' : det_state_dict['2'], '1' : det_state_dict['1'], 
              '2' : rec_state_dict['0'], '3' : rec_state_dict['1']}

# 要使用det的ckpt，因为需要united模式
syn_dkpt = {'state_dict' : state_dict, 'config' : det_ckpt['config']}

filename = "/root/CYX_Space/FOTS-master-failed/saved_model/best_united_model.pth.tar"
torch.save(syn_dkpt, filename)