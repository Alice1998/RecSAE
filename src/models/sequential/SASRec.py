# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" SASRec
Reference:
	"Self-attentive Sequential Recommendation"
	Kang et al., IEEE'2018.
Note:
	When incorporating position embedding, we make the position index start from the most recent interaction.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import json
import pandas as pd

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

from models.sae.sae import SAE


class SASRecBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		parser.add_argument('--num_heads', type=int, default=4,
							help='Number of attention heads.')
		return parser		

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.max_his = args.history_max
		self.num_layers = args.num_layers
		self.num_heads = args.num_heads
		self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
		self._base_define_params()
		self.apply(self.init_weights)

	def _base_define_params(self):
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
		self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

		self.transformer_block = nn.ModuleList([
			layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
									dropout=self.dropout, kq_same=False)
			for _ in range(self.num_layers)
		])

	def forward(self, feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # [batch_size, -1]
		history = feed_dict['history_items']  # [batch_size, history_max]
		lengths = feed_dict['lengths']  # [batch_size]
		batch_size, seq_len = history.shape

		valid_his = (history > 0).long()
		his_vectors = self.i_embeddings(history)

		# Position embedding
		# lengths:  [4, 2, 5]
		# position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
		position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
		pos_vectors = self.p_embeddings(position)
		his_vectors = his_vectors + pos_vectors

		# Self-attention
		causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
		attn_mask = torch.from_numpy(causality_mask).to(self.device)
		# attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
		for block in self.transformer_block:
			his_vectors = block(his_vectors, attn_mask)
		his_vectors = his_vectors * valid_his[:, :, None].float()

		his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
		# his_vector = his_vectors.sum(1) / lengths[:, None].float()
		# ↑ average pooling is shown to be more effective than the most recent embedding

		i_vectors = self.i_embeddings(i_ids)
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)

		u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = i_vectors

		return {'prediction': prediction.view(batch_size, -1), 'u_v': u_v, 'i_v':i_v}


class SASRec(SequentialModel, SASRecBase):
	reader = 'SeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']

	@staticmethod
	def parse_model_args(parser):
		parser = SASRecBase.parse_model_args(parser)
		return SequentialModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		SequentialModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = SASRecBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
class SASRecImpression(ImpressionSeqModel, SASRecBase):
	reader = 'ImpressionSeqReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']

	@staticmethod
	def parse_model_args(parser):
		parser = SASRecBase.parse_model_args(parser)
		return ImpressionSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ImpressionSeqModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return SASRecBase.forward(self, feed_dict)

TRAIN_MODE = 1
INFERENCE_MODE = 0
TEST_MODE = 2

class SASRec_SAE(SASRec):
	reader = 'SeqReader'
	runner = 'RecSAERunner'
	sae_extra_params = ['sae_lr','sae_batch_size','sae_k','sae_scale_size']

	@staticmethod
	def parse_model_args(parser):
		parser = SAE.parse_model_args(parser)
		parser = SASRec.parse_model_args(parser)
		return parser
	
	def __init__(self, args, corpus):
		SASRec.__init__(self, args, corpus)
		self.sae_module = SAE(args, self.emb_size)
		self.mode = "" # train, inference
		self.recsae_model_path = args.recsae_model_path

		self.epoch_users = None
		self.epoch_history_items = None
		self.epoch_embedding = None
		return
	

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

	def forward(self,feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # [batch_size, -1]
		history = feed_dict['history_items']  # [batch_size, history_max]
		lengths = feed_dict['lengths']  # [batch_size]
		batch_size, seq_len = history.shape

		valid_his = (history > 0).long()
		his_vectors = self.i_embeddings(history)

		# Position embedding
		# lengths:  [4, 2, 5]
		# position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
		position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
		pos_vectors = self.p_embeddings(position)
		his_vectors = his_vectors + pos_vectors

		# Self-attention
		causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
		attn_mask = torch.from_numpy(causality_mask).to(self.device)
		# attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
		for block in self.transformer_block:
			his_vectors = block(his_vectors, attn_mask)
		his_vectors = his_vectors * valid_his[:, :, None].float()


		his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
		# his_vector = his_vectors.sum(1) / lengths[:, None].float()
		# ↑ average pooling is shown to be more effective than the most recent embedding
		

		i_vectors = self.i_embeddings(i_ids)
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
		prediction_sae = []
		save_result_flag = {INFERENCE_MODE:False, TEST_MODE:True}
		if self.mode == INFERENCE_MODE or self.mode == TEST_MODE:
			# norms = torch.norm(his_vector, p=2, dim=1, keepdim=True)
			# sae_input = his_vector / norms
			# sae_output = self.sae_module(sae_input, save_result = save_result_flag[self.mode]) * norms
			sae_output = self.sae_module(his_vector, save_result = save_result_flag[self.mode])
			prediction_sae = (sae_output[:, None, :] * i_vectors).sum(-1)
			if self.mode == TEST_MODE:
				if self.epoch_users is None:
					self.epoch_users = feed_dict['user_id'].detach().cpu().numpy()
					self.epoch_history_items = history.detach().cpu().numpy()
					self.epoch_embedding = his_vector.detach().cpu().numpy()
					# import ipdbi
				else:
					self.epoch_users = np.concatenate((self.epoch_users, feed_dict['user_id'].detach().cpu().numpy()), axis=0)
					self.epoch_history_items = np.concatenate((self.epoch_history_items, history.detach().cpu().numpy()), axis=0)
					self.epoch_embedding = np.concatenate((self.epoch_embedding, his_vector.detach().cpu().numpy()), axis=0)
		elif self.mode == TRAIN_MODE:
			mask = (history > 0).bool()
			sae_input_ori = his_vectors[mask]
			if np.isnan(sae_input_ori[0,0].detach().cpu().data.numpy()):
				import ipdb;ipdb.set_trace()
			# norms = torch.norm(sae_input_ori, p=2, dim=1, keepdim=True)
			# sae_input = sae_input_ori / norms
			# sae_output = self.sae_module(sae_input, train_mode = True) * norms
			sae_output = self.sae_module(sae_input_ori, train_mode = True)
			# import ipdb;ipdb.set_trace()
			valid_indices = torch.cumsum(valid_his.view(-1), dim=0) - 1
			sae_output_v2 = sae_output[valid_indices[lengths - 1]]
			prediction_sae = (sae_output_v2[:, None, :] * i_vectors).sum(-1)
		else:
			raise ValueError('[SASRec_SAE-SAE] Undefined mode: {}.'.format(self.mode))

		u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = i_vectors
		return {'prediction': prediction.view(batch_size, -1), "prediction_sae": prediction_sae, 'u_v': u_v, 'i_v':i_v}
	
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
		# import ipdb;ipdb.set_trace()
		self.sae_module.epoch_activations['user_id'] = self.epoch_users
		self.sae_module.epoch_activations['history'] = self.epoch_history_items
		df = pd.DataFrame()
		df['user_id'] = self.epoch_users
		df['history'] = [np.trim_zeros(row, 'b').tolist() for row in self.epoch_history_items]
		df['indices'] = [x.tolist() for x in self.sae_module.epoch_activations['indices']]
		df['values'] =[x.tolist() for x in self.sae_module.epoch_activations['values']]
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