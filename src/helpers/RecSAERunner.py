import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from utils import utils
from models.BaseModel import BaseModel
import numpy as np
from scipy import stats
import json
from models.sae.sae import SAE
from typing import Dict, List
import pandas as pd
import gc

from helpers.BaseRunner import BaseRunner

def sig_test(x,y):
	t_stat, p_value = stats.ttest_ind(x, y)
	alpha = 0.05 
	if p_value < alpha:
		# print("两组数据之间存在显著性差异")
		return True
	else:
		# print("两组数据之间不存在显著性差异")
		return False
	
class RecSAERunner(BaseRunner):
	@staticmethod
	def parse_runner_args(parser):
		parser = BaseRunner.parse_runner_args(parser)
		parser.add_argument('--sae_lr', type=float, default=1e-4,
							help='SAE Learning rate.')
		parser.add_argument('--sae_batch_size', type=int, default=32,
							help='SAE batch size')
		parser.add_argument('--sae_train', type=int, default=0,
							help='train sae or evaluate RecSAE')
		parser.add_argument('--sae_baseline', type=int, default=0,
						help='To save baseline result or not')
		parser.add_argument('--result_data_path', type=str, default="",
							help='base path to save prediction list and RecSAE activations')
		parser.add_argument('--probe_position', type=str, default="",
							help='default, emb')
		
		return parser
	
	
	
	def __init__(self, args):
		BaseRunner.__init__(self,args)
		self.learning_rate = args.sae_lr
		self.sae_batch_size = args.sae_batch_size
		self.sae_train = args.sae_train
		self.sae_baseline = args.sae_baseline
		self.result_data_path = args.result_data_path
		self.probe_position = args.probe_position

	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		model.eval()

		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)

		# model.set_sae_mode("inference")
		# for key in ['dev','test']:
		# 	dev_result = self.evaluate(data_dict[key], self.topk[:1], self.metrics, prediction_label = "prediction")
		# 	dev_results.append(dev_result)
		# 	main_metric_results.append(dev_result[self.main_metric])
		# 	logging_str = '[Without SAE] {}=({})'.format(
		# 		key, utils.format_metric(dev_result))

		for epoch in range(self.epoch):
			self._check_time()
			gc.collect()
			torch.cuda.empty_cache()
			model.set_sae_mode("train")
			if self.probe_position=="":
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
			elif self.probe_position == 'emb':
				loss = self.fit_embedding(data_dict['train'], epoch=epoch + 1)
			else:
				raise ValueError(f"[SAE Runner] probe_position = {self.probe_position}")
			if np.isnan(loss):
				logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
				break
			training_time = self._check_time()
			dead_latent_ratio = model.get_dead_latent_ratio()
			logging_str = 'Epoch {:<5}loss={:<.4f}, dead_latent={:<.4f} [{:<3.1f} s]'.format(
				epoch + 1, loss, dead_latent_ratio, training_time)
			logging.info(logging_str)

			model.set_sae_mode("inference")
			# Record dev results
			dev_result = self.evaluate(data_dict['dev'], self.topk, self.metrics, prediction_label = "prediction_sae") # [self.main_topk]
			dev_results.append(dev_result)
			main_metric_results.append(dev_result[self.main_metric])
			dead_latent_ratio = model.get_dead_latent_ratio()
			logging_str = '[Dev] dead_latent={:<.4f}\ndev=({})'.format(
				 dead_latent_ratio, utils.format_metric(dev_result))

			# Test
			if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
				test_result = self.evaluate(data_dict['test'], self.topk, self.metrics, prediction_label = "prediction_sae")
				dead_latent_ratio = model.get_dead_latent_ratio()
				logging_str += '[Test] dead_latent={:<.4f}}\ntest=({})'.format(dead_latent_ratio, utils.format_metric(test_result))
			testing_time = self._check_time()
			logging_str += ' [{:<.1f} s]'.format(testing_time)

			if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
				model.save_model(model.recsae_model_path)
				logging_str += ' *'
			logging.info(logging_str)

			if self.early_stop > 0 and self.eval_termination(main_metric_results):
				logging.info("Early stop at %d based on dev result." % (epoch + 1))
				break
		
		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model(model.recsae_model_path)

	def fit_embedding(self, dataset, epoch = -1):
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		# dataset.actions_before_epoch()
		model.sae_module.set_decoder_norm_to_unit_norm()

		loss_lst = list()
		dataset.actions_before_epoch()
		dl = DataLoader(dataset, batch_size=self.sae_batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			item_id_list = batch['item_id']
			gt_items = item_id_list[:, 0]
			gt_item_embeddings = model.i_embeddings(gt_items)
			model.optimizer.zero_grad()
			batch_item_e_sae = model.sae_module(gt_item_embeddings, train_mode = True)
			# import ipdb;ipdb.set_trace()
			if epoch == 1:
				loss = model.sae_module.fvu
			else:
				loss = model.sae_module.fvu + model.sae_module.auxk_loss/32

			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		
		item_num = model.item_num
		item_embs = model.i_embeddings(torch.tensor([i for i in range(1,item_num)]).to(model.device))
		shuffle_indices = torch.randperm(item_embs.size(0)) 
		shuffled_item_embs = item_embs[shuffle_indices]     
		batch_data = torch.split(shuffled_item_embs, self.sae_batch_size)
		if len(batch_data) > 1 and batch_data[-1].shape[0] < self.sae_batch_size / 2:
			merged_batch = torch.cat([batch_data[-2], batch_data[-1]], dim=0)
			batch_data = batch_data[:-2] + (merged_batch,)
		for i, batch in enumerate(batch_data):
			model.optimizer.zero_grad()
			batch_item_e_sae = model.sae_module(batch, train_mode = True)
			
			if epoch == 1:
				loss = model.sae_module.fvu
			else:
				loss = model.sae_module.fvu + model.sae_module.auxk_loss/32

			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		loss = np.mean(loss_lst).item()
		if np.isnan(loss) or np.isposinf(loss):
			import ipdb;ipdb.set_trace()

		return loss
	
	def prediction_embedding(self, data_dict, epoch = -1):
		dataset = data_dict['test']
		model = dataset.model
		model.set_sae_mode("test")
		
		item_num = model.item_num
		item_embs = model.i_embeddings(torch.tensor([i for i in range(1,item_num)]).to(model.device))
		shuffle_indices = torch.randperm(item_embs.size(0)) 
		shuffled_item_embs = item_embs[shuffle_indices]     
		batch_data = torch.split(shuffled_item_embs, self.sae_batch_size)
		for i, batch in enumerate(batch_data):
			batch_item_e_sae = model.sae_module(batch, save_result = True)

		model_path = self.result_data_path + "_activation.csv"
		emb_path = self.result_data_path + "_emb.npy"
		dataset.model.save_epoch_result(dataset,path = model_path, emb_path = emb_path)
		logging.info(f'[RecSAE Runner] save activation data\n{model_path}')
		return
	
	def fit(self, dataset, epoch = -1):
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()
		model.sae_module.set_decoder_norm_to_unit_norm()
		
		loss_lst = list()
		dl = DataLoader(dataset, batch_size=self.sae_batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			if batch['user_id'].shape[0] == 1:
				continue
			batch = utils.batch_to_gpu(batch, model.device)
			# randomly shuffle the items to avoid models remembering the first item being the target
			item_ids = batch['item_id']
			# for each row (sample), get random indices and shuffle the original items
			indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
			batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

			model.optimizer.zero_grad()
			out_dict = model(batch)

			if epoch == 1:
				loss = model.sae_module.fvu
			else:
				loss = model.sae_module.fvu + model.sae_module.auxk_loss/32

			loss.backward()
			model.optimizer.step()
			loss_value = loss.detach().cpu().data.numpy()
			if np.isnan(loss_value) or np.isinf(loss_value):
				import ipdb;ipdb.set_trace()
			loss_lst.append(loss_value)
		loss = np.mean(loss_lst).item()
		if np.isnan(loss) or np.isinf(loss):
			import ipdb;ipdb.set_trace()

		return loss
	
	def print_res(self, dataset: BaseModel.Dataset, prediction_label = "prediction", save_result = False) -> str:
		if save_result: # sae_train 0 and test set
			dataset.model.set_sae_mode("test")
		else:
			dataset.model.set_sae_mode("inference")
		result = BaseRunner.print_res(self,dataset, prediction_label=prediction_label)
		# emb saved in function prediction_embedding
		if save_result:
			if self.sae_baseline:
				model_path = self.result_data_path + "_activation_baseline.csv"
				rec_model_path = self.result_data_path + "_recmodel_baseline.npy"
				np.save(rec_model_path,dataset.model.sae_module.rec_model_activations)
			else:
				model_path = self.result_data_path + "_activation.csv"
			emb_path = self.result_data_path + "_emb.npy"
			dataset.model.save_epoch_result(dataset,path = model_path, emb_path = emb_path)
			logging.info(f'[RecSAE Runner] save activation data\n{model_path}')
		return result

