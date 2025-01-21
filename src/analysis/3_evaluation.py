from sentence_transformers import SentenceTransformer

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import json
from sklearn.preprocessing import StandardScaler
import random


def load_data(activation_file, prediction_file):
	act_df = pd.read_csv(activation_file,delimiter = '\t')
	pred_df = pd.read_csv(prediction_file, delimiter = '\t')

	for col in act_df.columns.tolist():
		if act_df [col].dtype == 'object':
			act_df [col] = act_df [col].apply(lambda v:eval(v))
	for col in pred_df.columns.tolist():
		if pred_df [col].dtype == 'object':
			pred_df [col] = pred_df [col].apply(lambda v:eval(v))
	# pred_df['history'] = act_df['history']
	pred_df['index_id'] = pred_df.index
	act_df['index_id'] = [i for i in range(len(act_df))]
	act_df = act_df.explode(['indices', 'values']).reset_index(drop=True)
	act_df['indices'] = act_df['indices'].astype(int)
	act_df['values'] = act_df['values'].astype(float)
	return act_df, pred_df

def getItemTitle(row, dataset):
	if dataset == "LastFM" or dataset == "LastFM_8k":
		if len(row['max_tag']) > 0:
			title = f"{row['title']}, in album {row['album_name']}, by {row['artist_name']} ({row['max_tag']})"
		else:
			title = f"{row['title']}, in album {row['album_name']}, by {row['artist_name']}"
	elif dataset == "ML_1MTOPK":
		category_list = row['genres'].split('|')
		categories = ', '.join(category_list)
		title = f"{row['title']} ({categories})"
	else:
		title = row['title']
	return title


def metric_cov():
	return

def analysis():
	return

def cluster():

	return



def prepare_data(base_model, dataset_name):
	# base_model = "SASRec"
	# base_model = "BPRMF"
	# base_model = "TiMiRec"

	sae_k = 8
	sae_scale_size = 16

	# dataset_name = "Grocery_and_Gourmet_Food"
	# dataset_name = "ML_1MTOPK"
	# dataset_name = "LastFM_8k"

	SASRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"1e-05",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"0.0001",8:"5e-05",16:"0.0001"}}	
	SASRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:4,16:16},\
			   	"LastFM_8k":{4:4,8:4,16:4}}
	
	bprmf_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"5e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05"}}
	bprmf_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:16,16:16},\
			   	"LastFM_8k":{4:16,8:4,16:4}}
	
	LightGCN_lr_dict = {"Grocery_and_Gourmet_Food":{8:"5e-05"},\
				"ML_1MTOPK":{8:"1e-05"},\
			   	"LastFM_8k":{8:"5e-05"}
				   }
	LightGCN_batch_dict = {"Grocery_and_Gourmet_Food":{8:4},\
				"ML_1MTOPK":{8:4},\
			   	"LastFM_8k":{8:16}
				}
	
	TiMiRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"0.0001"},\
				"ML_1MTOPK":{4:"0.0001",8:"1e-05",16:"5e-05"},\
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001"}}
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16}}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"TiMiRec":TiMiRec_lr_dict,"LightGCN":LightGCN_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,'TiMiRec':TiMiRec_batch_dict,"LightGCN":LightGCN_batch_dict}
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]

	if base_model == 'SASRec' or base_model == 'TiMiRec':
		file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test-history.csv"
	elif base_model == "BPRMF" or base_model == "LightGCN":
		file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test.csv"
	pred_df = pd.read_csv(file_path,delimiter="\t")
	pred_df['index_id'] = pred_df.index
	# pred_df.set_index('index_id',inplace=True)
	for col in pred_df.columns.tolist():
		if pred_df [col].dtype == 'object':
			pred_df [col] = pred_df [col].apply(lambda v:eval(v))
	pred_df['pred_item'] = pred_df['rec_items'].apply(lambda v:v[0])

	if base_model == "SASRec":
		file_path = f"../../log/SASRec_SAE/result_file/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == 'BPRMF':
		file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == 'TiMiRec':
		file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == 'DirectAU':
		file_path = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_recmodel_baseline.npy"
	elif base_model == "LightGCN":
		file_path = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_recmodel_baseline.npy"
	
	rec_model_np = np.load(file_path)
	rec_model_np = rec_model_np/rec_model_np.max()

	rec_model_dict = {}
	for i in range(rec_model_np.shape[1]): 
		sorted_indices = np.argsort(rec_model_np[:, i])
		item_act_dict = {"index_id":[],"item_id":[],"act_value":[]}
		index = 0
		while len(item_act_dict['item_id']) < 5:
			rec_index = int(sorted_indices[-index-1])
			item_id = pred_df.loc[rec_index,'pred_item']
			if item_id not in item_act_dict['item_id']:
				value = float(rec_model_np[rec_index, i])
				item_act_dict['index_id'] .append(rec_index)
				item_act_dict['item_id'].append(int(item_id))
				item_act_dict['act_value'].append(value)
			index+=1
		rec_model_dict[i] = item_act_dict
	# import ipdb;ipdb.set_trace()

	result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_evaluate_method=recModel.jsonl"
	with open(result_file,'w') as f:
		for key, value in rec_model_dict.items():
			json.dump({key: value}, f)
			f.write("\n")


	for baseline in [0, 1]:
		all_data = dict()
		# print(f"[start]\nbaseline {baseline}")
		if baseline == 1:
			method = "_baseline"
		else:
			method = ""
		if base_model == "SASRec":
			activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
			prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		elif base_model == 'BPRMF':
			activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
			prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		elif base_model == 'TiMiRec':
			activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
			prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		elif base_model == 'DirectAU':
			activation_file = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_activation{method}.csv"
			prediction_file = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_prediction{method}.csv"
		elif base_model == "LightGCN":
			activation_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_activation{method}.csv"
			prediction_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_prediction{method}.csv"

		act_df, _ = load_data(activation_file, prediction_file)

		act_df = act_df[act_df['values']>0]
		act_df = act_df.sample(frac = 1)
		categories, bins = pd.qcut(act_df['values'], q=10, retbins=True,labels = range(1,11))
		act_df['act_level'] = categories.astype(int)

		act_df.rename({"indices":'latent_id'},axis = 1,inplace = True)
		# pred_df['pred_item'] = pred_df['rec_items'].apply(lambda v:v[0])


		act_df = pd.merge(act_df, pred_df[['index_id','pred_item']], on = 'index_id', how = 'left')
		act_df.sort_values(['latent_id','values'],ascending=[True,False],inplace=True)
		act_df['values'] = act_df['values']/act_df['values'].max()

		latentDict = dict(list(act_df.groupby('latent_id')))

		# all_data[baseline] = latentDict
		for latent_id in latentDict:
			df = latentDict[latent_id]
			item_act_dict = {"index_id":[],"item_id":[],"act_value":[]}
			for index,row in df.iterrows():
				if len(item_act_dict['item_id']) == 5:
					break
				item_id = row['pred_item']
				if item_id not in item_act_dict['item_id']:
					value = row['values']
					rec_index = int(row['index_id'])
					item_act_dict['index_id'].append(rec_index)
					item_act_dict['act_value'].append(value)
					item_act_dict['item_id'].append(item_id)
			# import ipdb;ipdb.set_trace()
			all_data[latent_id] = item_act_dict

		result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_evaluate_method={baseline}.jsonl"
		with open(result_file,'w') as f:
			for key, value in all_data.items():
				json.dump({key: value}, f)
				f.write("\n")



	return

def analysis(base_model, dataset_name):

	# base_model = "SASRec"
	# base_model = "BPRMF"
	# base_model = "TiMiRec"
	# base_model = "LightGCN"

	sae_k = 8
	sae_scale_size = 16

	# dataset_name = "Grocery_and_Gourmet_Food"
	# dataset_name = "ML_1MTOPK"
	# dataset_name = "LastFM_8k"

	SASRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"1e-05",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"0.0001",8:"5e-05",16:"0.0001"}}	
	SASRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:4,16:16},\
			   	"LastFM_8k":{4:4,8:4,16:4}}
	
	bprmf_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"5e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05"}}
	bprmf_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:16,16:16},\
			   	"LastFM_8k":{4:16,8:4,16:4}}
	
	TiMiRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"0.0001"},\
				"ML_1MTOPK":{4:"0.0001",8:"1e-05",16:"5e-05"},\
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001"}}
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16}}
	
	LightGCN_lr_dict = {"Grocery_and_Gourmet_Food":{8:"5e-05"},\
				"ML_1MTOPK":{8:"1e-05"},\
			   	"LastFM_8k":{8:"5e-05"}
				   }
	LightGCN_batch_dict = {"Grocery_and_Gourmet_Food":{8:4},\
				"ML_1MTOPK":{8:4},\
			   	"LastFM_8k":{8:16}
				}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"TiMiRec":TiMiRec_lr_dict,"LightGCN":LightGCN_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,'TiMiRec':TiMiRec_batch_dict,"LightGCN":LightGCN_batch_dict}
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]

	result_file = {"recModel":f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_evaluate_method=recModel.jsonl",\
				'Init':f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_evaluate_method=1.jsonl",\
				"RecSAE":f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_evaluate_method=0.jsonl"}
	confidence_score_dict = {
				"recModel":f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_result_v5_recModel.csv",\
				"Init":f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_result_v5_baseline.csv",\
				"RecSAE":f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_result_v5.csv"
		}
	
	if base_model == "SASRec":
		file_path = f"../../log/SASRec_SAE/result_file/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_emb.npy"
	elif base_model == 'BPRMF':
		file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_emb.npy"
	elif base_model == 'TiMiRec':
		file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_emb.npy"
	elif base_model == 'DirectAU':
		file_path = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_emb.npy"
	elif base_model == "LightGCN":
		file_path = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_emb.npy"
	
	# emb_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_emb.npy"
	emb_np = np.load(file_path)

	

	for key in result_file:
		with open(result_file[key],'r') as f:
			all_data_read = {}
			for line in f:
				data = json.loads(line.strip())  # 使用 json.loads 解析每行的 JSON 数据
				all_data_read.update(data) 
				# print(data)  # 输出每行的字典
			# import ipdb;ipdb.set_trace()
		if key != 'recModel':
			confidence_df = pd.read_csv(confidence_score_dict[key])
			confidence_df.set_index('latent_id',inplace=True)

		result = []
		for latent_id in all_data_read:
			int_id = int(latent_id)
			if key!= 'recModel' and (int_id not in confidence_df.index or confidence_df.loc[int_id,'confidence_score']<0.8):
				continue
			single_info = all_data_read[latent_id]
			# print(single_info['index_id'])
			embeddings = [emb_np[i] for i in single_info['index_id'][:2]]
			length = len(embeddings)
			if length == 1:
				continue
			value_all = 0
			matrix = cosine_similarity(embeddings)
			counter = 0
			for i in range(length-1):
				for j in range(i+1,length):
					value_all += float(matrix[i,j].item())
					counter += 1
			value = value_all / counter
			# import ipdb;ipdb.set_trace()
			# value = float(cosine_similarity(embeddings)[0,1].item())
			# import ipdb;ipdb.set_trace()
			result.append(value)
			# print(value)
		print(np.mean(result))

	# counter_list = []
	# for key in result_file:
	# 	if key != 'recModel':
	# 		confidence_df = pd.read_csv(confidence_score_dict[key])
	# 		confidence_df.set_index('latent_id',inplace=True)

	# 	with open(result_file[key],'r') as f:
	# 		all_data_read = {}
	# 		for line in f:
	# 			data = json.loads(line.strip())  # 使用 json.loads 解析每行的 JSON 数据
	# 			all_data_read.update(data) 
	# 	emb_list = []
	# 	count_latent = 0
	# 	for latent_id in all_data_read:
	# 		int_id = int(latent_id)
	# 		if key!= 'recModel' and (int_id not in confidence_df.index or confidence_df.loc[int_id,'confidence_score']<0.8):
	# 			continue
	# 		count_latent += 1
	# 		single_info = all_data_read[latent_id]
	# 		index = single_info['index_id'][0]
	# 		emb = emb_np[index]
	# 		emb_list.append(emb)
	# 	matrix = cosine_similarity(emb_list)
	# 	length = len(emb_list)
	# 	value_all = 0
	# 	counter = 0
	# 	for i in range(length-1):
	# 		for j in range(i+1,length):
	# 			value_all += float(matrix[i,j].item())
	# 			counter += 1
	# 	value = value_all / counter
	# 	print(value)
	# 	# print(count_latent)
	# 	counter_list.append(count_latent)
	# # print(counter_list)









	return
	



if __name__=='__main__':


	for dataset_name in ["Grocery_and_Gourmet_Food",'ML_1MTOPK','LastFM_8k']:

		for base_model in ['BPRMF',"LightGCN",'SASRec','TiMiRec']:

			analysis(dataset_name= dataset_name,base_model=base_model)
	