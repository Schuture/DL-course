import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import my_collate_fn, custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, start_epoch, epoch_iter, interval):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, collate_fn=my_collate_fn, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(pretrained=False)
	model.load_state_dict(torch.load("/root/CYX_Space/EAST-master/pths/model_epoch_695.pth"))
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)

	for epoch in range(start_epoch, epoch_iter):
		model.train()
		scheduler.step(epoch)
		print('\nLearning rate this epoch is {}\n'.format(scheduler.get_lr()[0]))
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			try:
				start_time = time.time()
				img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
				pred_score, pred_geo = model(img)
				loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
				
				epoch_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
				  epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
			except:
				print('Meet exception this iteration.')
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
	train_img_path = os.path.abspath('../data/train/img')
	train_gt_path  = os.path.abspath('../data/train/gt')
	pths_path      = './pths'
	batch_size     = 8
	lr             = 1e-3
	num_workers    = 16
	epoch_iter     = 800
	save_interval  = 5
	start_epoch = 697 # 从start_epoch+1开始训练
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, start_epoch, epoch_iter, save_interval)	
	
