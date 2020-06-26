''' 生成测试集上的rbox坐标预测 '''

import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path): # 将之前输出的结果删掉
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag: # 只评估，不保存结果
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = "/root/CYX_Space/EAST-master/pths/model_epoch_785.pth"
	test_img_path = "/root/CYX_Space/data/test/img/"
	submit_path = './submit' # 用于竞赛提交结果, .txt
	eval_model(model_name, test_img_path, submit_path, save_flag=True)
