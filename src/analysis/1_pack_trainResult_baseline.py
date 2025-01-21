import pandas as pd
import random
import json
import numpy as np


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

if __name__ == "__main__":
	
	method = ""

	# base_model = "SASRec"
	# base_model = "BPRMF"
	# base_model = "TiMiRec"
	base_model = "LightGCN"

	sae_k = 8
	sae_scale_size = 16
	
	dataset_name = "Grocery_and_Gourmet_Food"
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
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001"}
				   }
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16}
				}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"LightGCN":LightGCN_lr_dict,"TiMiRec":TiMiRec_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,"LightGCN":LightGCN_batch_dict,'TiMiRec':TiMiRec_batch_dict}
	
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]



	if base_model == "SASRec":
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == 'BPRMF':
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		file_path = f"../../log/BPRMF_SAE/result_file/BPRMF_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == "TiMiRec":
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
		file_path = f"../../log/{base_model}_SAE/result_file/TiMiRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	elif base_model == "LightGCN":
		activation_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_activation{method}.csv"
		prediction_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_prediction{method}.csv"
		file_path = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	act_df, _ = load_data(activation_file, prediction_file)

	# file_path = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_recmodel_baseline.npy"
	rec_model_np = np.load(file_path)

	file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test.csv"
	prediction_df = pd.read_csv(file_path,delimiter="\t")

	for col in prediction_df.columns.tolist():
		if prediction_df [col].dtype == 'object':
			prediction_df [col] = prediction_df [col].apply(lambda v:eval(v))


	

	# print(rec_model_np)
	# (14681, 64)
	# quantiles = np.percentile(rec_model_np, np.linspace(0, 100, 11), axis=0)
	# rec_levels = np.digitize(rec_model_np, quantiles[1:], right=True) - 1  # 等级从0开始

	quantiles = np.percentile(rec_model_np, np.linspace(0, 100, 11), axis=0)
	rec_levels = np.zeros_like(rec_model_np, dtype=int)
	for i in range(rec_model_np.shape[1]):
		quantiles_1d = quantiles[:, i]
		rec_levels[:, i] = np.digitize(rec_model_np[:, i], quantiles_1d[1:], right=True) - 1

	SELECT_NUM = 5
	final_selection_dict = {}
	index_length = rec_model_np.shape[0]
	index_id_list = range(index_length)
	for i in range(rec_model_np.shape[1]): 
		latent_result = dict()
		sorted_indices = np.argsort(rec_model_np[:, i])
		high_activation_list = []
		high_act_value_list = []
		for index in range(10):
			rec_index = sorted_indices[-index-1]
			# item_id = prediction_df.loc[rec_index,'pred_items']
			high_activation_list.append(rec_index)
			high_act_value_list.append(rec_levels[rec_index, i])

		all_numbers = range(10)
		selected_numbers = random.sample(all_numbers, SELECT_NUM)
		unselected_numbers = [num for num in all_numbers if num not in selected_numbers]
		selection = {'train':selected_numbers,'test':unselected_numbers}
		for key in ['train','test']:
			latent_result[key] = [high_activation_list[index] for index in selection[key]]
			latent_result[f"{key}_acts"] = [high_act_value_list[index] for index in selection[key]]
		
		selected_index_ids = np.where(rec_levels[:, i] >= 5)[0].tolist()
		remaining_users = [index_id for index_id in index_id_list if index_id not in selected_index_ids]
		random_selection = random.sample(remaining_users, SELECT_NUM)
		latent_result['random'] = random_selection
		final_selection_dict[i] = latent_result
	
	def convert(obj):
		if isinstance(obj, np.int64):
			return int(obj)  # 将 int64 转换为 int
		raise TypeError(f"Type {type(obj)} not serializable")


	result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__trainTest_info_norm_haveTestAll_recModel.jsonl"
	with open(result_file,'w') as f:
		for key, value in final_selection_dict.items():
			json.dump({key: value}, f, default = convert)
			f.write("\n")