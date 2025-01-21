# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BPRMF
Reference:
	"Bayesian personalized ranking from implicit feedback"
	Rendle et al., UAI'2009.
CMD example:
	python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch.nn as nn

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

from models.sae.sae import SAE
import logging
import numpy as np
import pandas as pd
import torch

class BPRMFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		cf_u_vectors = self.u_embeddings(u_ids)
		cf_i_vectors = self.i_embeddings(i_ids)

		prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = cf_i_vectors
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class BPRMF(GeneralModel, BPRMFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict =  BPRMFBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}

class BPRMFImpression(ImpressionModel, BPRMFBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return BPRMFBase.forward(self, feed_dict)
	

TRAIN_MODE = 1
INFERENCE_MODE = 0
TEST_MODE = 2

class BPRMF_SAE(BPRMF):
	runner = "RecSAERunner"
	sae_extra_params = ['sae_lr','sae_batch_size','sae_k','sae_scale_size']


	@staticmethod
	def parse_model_args(parser):
		parser = SAE.parse_model_args(parser)
		parser = BPRMF.parse_model_args(parser)
		return parser
	
	def __init__(self, args, corpus):
		BPRMF.__init__(self, args, corpus)
		self.sae_module = SAE(args, self.emb_size)
		self.mode = ""
		self.recsae_model_path = args.recsae_model_path

		self.epoch_users = None
		self.epoch_history_items = None
		self.epoch_embedding = None

	def set_sae_mode(self, mode):
		if mode == 'train':
			self.mode = TRAIN_MODE
		elif mode == 'inference':
			self.mode = INFERENCE_MODE
		elif mode == 'test':
			self.mode = TEST_MODE
		else:
			raise ValueError(f"[SASRec-SAE] mode ERROR!!! mode = {mode}")
	
	def get_dead_latent_ratio(self):
		return self.sae_module.get_dead_latent_ratio(need_update = self.mode)

	# def forward(self, feed_dict):
	# 	self.check_list = []
	# 	u_ids = feed_dict['user_id']  # [batch_size]
	# 	i_ids = feed_dict['item_id']  # [batch_size, -1]

	# 	cf_u_vectors = self.u_embeddings(u_ids)
	# 	cf_i_vectors = self.i_embeddings(i_ids)

	# 	batch_size = feed_dict['batch_size']
	# 	save_result_flag = {TRAIN_MODE:False,INFERENCE_MODE:False, TEST_MODE:True}
	# 	batches = torch.split(cf_i_vectors[0,1:,:], batch_size)
	# 	sae_output_list = []
	# 	for i, batch in enumerate(batches):
	# 		batch_item_e_sae = self.sae_module(batch, save_result = save_result_flag[self.mode])
	# 		sae_output_list.append(batch_item_e_sae)
	# 	candidate_emb_sae = torch.cat(sae_output_list, dim = 0).unsqueeze(0).expand(batch_size,-1,-1)

	# 	ground_truth_emb = cf_i_vectors[:,0,:]
	# 	ground_truth_emb_sae= self.sae_module(ground_truth_emb, save_result = save_result_flag[self.mode])
	# 	ground_truth_emb_sae = ground_truth_emb_sae.unsqueeze(1)
	
	# 	cf_i_vectors_sae = torch.cat((ground_truth_emb_sae, candidate_emb_sae), dim=1) 
	# 	self.epoch_history_items = []
	# 	self.epoch_users = []
	# 	sae_prediction = (cf_u_vectors[:, None, :] * cf_i_vectors_sae).sum(dim=-1)
	# 	sae_output_result = sae_prediction.view(feed_dict['batch_size'], -1)


	# 	prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
	# 	# u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
	# 	# i_v = cf_i_vectors
	# 	return {'prediction': prediction.view(feed_dict['batch_size'], -1), "prediction_sae":sae_output_result}
	
	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		cf_u_vectors = self.u_embeddings(u_ids)
		cf_i_vectors = self.i_embeddings(i_ids)

		save_result_flag = {INFERENCE_MODE:False, TEST_MODE:True}
		if self.mode == INFERENCE_MODE or self.mode == TEST_MODE:
			sae_output = self.sae_module(cf_u_vectors, save_result = save_result_flag[self.mode])
			if self.mode == TEST_MODE:
				if self.epoch_users is None:
					self.epoch_users = feed_dict['user_id'].detach().cpu().numpy()
					self.epoch_embedding = cf_u_vectors.detach().cpu().numpy()
				else:
					self.epoch_users = np.concatenate((self.epoch_users, feed_dict['user_id'].detach().cpu().numpy()), axis=0)
					self.epoch_embedding = np.concatenate((self.epoch_embedding, cf_u_vectors.detach().cpu().numpy()), axis=0)
		elif self.mode == TRAIN_MODE:
			sae_output = self.sae_module(cf_u_vectors, train_mode = True)
		prediction_sae = (sae_output[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = cf_i_vectors
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), "prediction_sae":prediction_sae.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path

		if model_path == self.model_path:
			state_dict = torch.load(model_path)
			self.load_state_dict(state_dict, strict = False)
			for name, param in self.named_parameters():
				if name in state_dict:
					param.requires_grad = False
		else:
			self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)
		return

	def save_epoch_result(self, dataset, path = None, emb_path = None):
		# self.sae_module.epoch_activations['user_id'] = self.epoch_users
		# self.sae_module.epoch_activations['history'] = self.epoch_history_items
		df = pd.DataFrame()
		# df['user_id'] = self.epoch_users
		# if self.sae_module.epoch_activations['history'] != None:
		# 	df['history'] = [np.trim_zeros(row, 'b').tolist() for row in self.epoch_history_items]
		df['user_id'] = self.epoch_users
		df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
		df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]
		# if len(df) != self.item_num-1:
		# 	raise ValueError('[BPRMF] save result != self.item_num')
		# df['item_id'] = [i for i in range(1,self.item_num)]
		df.to_csv(path,sep = "\t",index=False)
		# with open(path,'w') as f:
		# 	f.write(json.dumps(self.sae_module.epoch_activations))

		np.save(emb_path, self.epoch_embedding)
		logging.info('save emb to '+ emb_path)

		self.sae_module.epoch_activations = {"indices": None, "values": None} 
		self.epoch_users = None
		self.epoch_history_items = None
		self.epoch_embedding = None
		return