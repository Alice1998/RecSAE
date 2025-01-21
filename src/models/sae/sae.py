import torch
import numpy as np
import json
import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging


class SAE(nn.Module):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--sae_k', type=int, default=32,
							help='top k activation')
		parser.add_argument('--sae_scale_size', type=int, default=32,
							help='scale size')
		parser.add_argument('--recsae_model_path', type=str, default='',
							help='Model save path.')
		return parser
	
	def __init__(self,args,d_in):
		super(SAE, self).__init__()

		self.k = args.sae_k
		self.scale_size = args.sae_scale_size

		self.device = args.device
		self.dtype = torch.float32

		self.d_in = d_in
		self.hidden_dim = d_in * self.scale_size

		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))

		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_activations = {"indices": None, "values": None} 

		self.sae_baseline = args.sae_baseline
		self.rec_model_activations = None

		return
	
	def reset_module_weight(self):
		self.encoder.reset_parameters()
		shape = self.W_dec.shape
		self.W_dec = nn.Parameter(torch.empty(shape, device=self.device))
		shape = self.b_dec.shape
		self.b_dec =  nn.Parameter(torch.empty(shape, device=self.device))

	def get_dead_latent_ratio(self, need_update = 0):
		ans =  1 - len(self.activate_latents)/self.hidden_dim
		# only update training situation for auxk_loss
		if need_update:
			# logging.info("[SAE] update previous activated Latent here")
			self.previous_activate_latents = torch.tensor(list(self.activate_latents)).to(self.device)
		self.activate_latents = set()
		return ans


	def set_decoder_norm_to_unit_norm(self):
		assert self.W_dec is not None, "Decoder weight was not initialized."
		eps = torch.finfo(self.W_dec.dtype).eps
		norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
		self.W_dec.data /= norm + eps

	

	def topk_activation(self, x, save_result):
		topk_values, topk_indices = torch.topk(x, self.k, dim=1)
		self.activate_latents.update(topk_indices.cpu().numpy().flatten())

		if save_result:
			# import ipdb;ipdb.set_trace()
			if self.epoch_activations["indices"] is None:
				self.epoch_activations["indices"] = topk_indices.detach().cpu().numpy()
				self.epoch_activations["values"] = topk_values.detach().cpu().numpy()
			else:
				self.epoch_activations["indices"] = np.concatenate((self.epoch_activations["indices"], topk_indices.detach().cpu().numpy()), axis=0)
				self.epoch_activations["values"] = np.concatenate((self.epoch_activations["values"], topk_values.detach().cpu().numpy()), axis=0)

		sparse_x = torch.zeros_like(x)
		sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
		return sparse_x
	

	def forward(self, x, train_mode = False, save_result = False):
		if self.sae_baseline:
			if self.rec_model_activations is None:
				self.rec_model_activations = x.detach().cpu().numpy()
			else:
				self.rec_model_activations = np.concatenate((self.rec_model_activations,x.detach().cpu().numpy()), axis = 0)

		sae_in = x - self.b_dec
		pre_acts = nn.functional.relu( self.encoder(sae_in) )
		z = self.topk_activation(pre_acts, save_result = save_result)
		x_reconstructed = z@self.W_dec + self.b_dec

		e = x_reconstructed - x
		total_variance = (x - x.mean(0)).pow(2).sum()
		self.fvu = e.pow(2).sum() / total_variance

		if train_mode:
			# first epoch, do not have dead latent info
			if (self.previous_activate_latents) is None:
				self.auxk_loss = 0.0
				return x_reconstructed
			num_dead = self.hidden_dim - len(self.previous_activate_latents)
			k_aux = x.shape[-1] // 2 
			if num_dead == 0:
				self.auxk_loss = 0.0
				return x_reconstructed
			# if k_aux > num_dead:
			# 	self.auxk_loss = 0.0
			# 	return x_reconstructed
			scale = min(num_dead / k_aux, 1.0)
			k_aux = min(k_aux, num_dead)
			dead_mask = torch.isin(torch.arange(pre_acts.shape[-1]).to(self.device), self.previous_activate_latents,invert=True)
			auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
			auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
			e_hat = torch.zeros_like(auxk_latents)
			e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
			e_hat = e_hat @ self.W_dec + self.b_dec

			auxk_loss = (e_hat - e).pow(2).sum()
			self.auxk_loss = scale * auxk_loss / total_variance

		return x_reconstructed

		