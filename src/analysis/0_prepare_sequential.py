import pandas as pd
import random
import json


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

	file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test.csv"
	prediction = pd.read_csv(file_path,delimiter="\t")
	prediction['history'] = act_df['history']

	file_path = f"../../log/{base_model}/{base_model}__{dataset_name}__0__lr=0/rec-{base_model}-test-history.csv"
	prediction.to_csv(file_path,sep = "\t")

	act_df['index_id'] = [i for i in range(len(act_df))]
	act_df = act_df.explode(['indices', 'values']).reset_index(drop=True)
	act_df['indices'] = act_df['indices'].astype(int)
	act_df['values'] = act_df['values'].astype(float)
	return act_df, pred_df


if __name__ == "__main__":

	# base_model = "SASRec"
	# base_model = "BPRMF"
	base_model = "TiMiRec"

	sae_k = 4
	sae_scale_size = 16
	
	method = ""

	# dataset_name = "Grocery_and_Gourmet_Food"
	# dataset_name = "ML_1MTOPK"
	dataset_name = "LastFM_8k"

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
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001"}
				   }
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16}
				}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"TiMiRec":TiMiRec_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,'TiMiRec':TiMiRec_batch_dict}
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]



	if base_model == "SASRec":
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
	elif base_model == 'BPRMF':
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
	elif base_model == 'TiMiRec':
		activation_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_activation{method}.csv"
		prediction_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__attn_size=8__K=6__temp=1.0__add_pos=1__add_trm=1__n_layers=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}_prediction{method}.csv"
	
	act_df, pred_df = load_data(activation_file, prediction_file)
