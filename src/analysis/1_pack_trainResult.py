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
	act_df['index_id'] = [i for i in range(len(act_df))]
	act_df = act_df.explode(['indices', 'values']).reset_index(drop=True)
	act_df['indices'] = act_df['indices'].astype(int)
	act_df['values'] = act_df['values'].astype(float)
	return act_df, pred_df



if __name__ == "__main__":
	method = ""
	# method = "_baseline"
	# method = "_recModel"

	# base_model = "BPRMF"
	# base_model = "SASRec"
	# base_model = "TiMiRec"
	# base_model = "DirectAU"
	base_model = "LightGCN"

	sae_k = 16
	sae_scale_size = 16

	# dataset_name = "Grocery_and_Gourmet_Food"
	# dataset_name = "ML_1MTOPK"
	dataset_name = "LastFM_8k"

	SASRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"1e-05",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"0.0001",8:"5e-05",16:"0.0001",32:"5e-05"}}
	SASRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:4,16:16},\
			   	"LastFM_8k":{4:4,8:4,16:4,32:4}}
	
	bprmf_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"5e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"0.0001",16:"0.0001"},\
			   	"LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05",32:"1e-05"}}
	bprmf_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:16,16:16},\
			   	"LastFM_8k":{4:16,8:4,16:4,32:16}}
	
	DirectAU_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"5e-05",16:"1e-05"},\
				"ML_1MTOPK":{4:"0.0001",8:"5e-05",16:"5e-05"},\
			   	# "LastFM_8k":{4:"5e-05",8:"1e-05",16:"1e-05"}
				}
	DirectAU_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:4,8:16,16:16},\
			   	# "LastFM_8k":{4:16,8:4,16:4}
				}
	
	LightGCN_lr_dict = {"Grocery_and_Gourmet_Food":{4:"0.0001",8:"5e-05",16:"0.0001"},\
				"ML_1MTOPK":{8:"1e-05"},\
			   	"LastFM_8k":{4:"5e-05",8:"5e-05",16:"0.0001",32:"5e-05"}
				   }
	LightGCN_batch_dict = {"Grocery_and_Gourmet_Food":{4:4,8:4,16:16},\
				"ML_1MTOPK":{8:4},\
			   	"LastFM_8k":{4:4,8:16,16:4,32:16}
				}
	
	TiMiRec_lr_dict = {"Grocery_and_Gourmet_Food":{4:"5e-05",8:"1e-05",16:"0.0001"},\
				"ML_1MTOPK":{4:"0.0001",8:"1e-05",16:"5e-05"},\
			   	"LastFM_8k":{4:"0.0001",8:"0.0001",16:"0.0001",32:"0.0001"}
				   }
	TiMiRec_batch_dict = {"Grocery_and_Gourmet_Food":{4:16,8:4,16:4},\
				"ML_1MTOPK":{4:16,8:4,16:16},\
			   	"LastFM_8k":{4:16,8:16,16:16,32:16}
				}
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"LightGCN":LightGCN_lr_dict,"TiMiRec":TiMiRec_lr_dict,'DirectAU':DirectAU_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,"LightGCN":LightGCN_batch_dict,'TiMiRec':TiMiRec_batch_dict,'DirectAU':DirectAU_batch_dict}
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
	elif base_model == 'DirectAU':
		activation_file = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_activation{method}.csv"
		prediction_file = f"../../log/DirectAU_SAE/result_file/DirectAU_SAE__{dataset_name}__0__lr=0.001__l2=1e-06__emb_size=64__gamma=0.3__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_prediction{method}.csv"
	elif base_model == "LightGCN":
		activation_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_activation{method}.csv"
		prediction_file = f"../../log/LightGCN_SAE/result_file/LightGCN_SAE__{dataset_name}__0__lr=0.001__l2=1e-08__emb_size=64__n_layers=3__batch_size=256__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size=16_prediction{method}.csv"
	act_df, pred_df = load_data(activation_file, prediction_file)
	# act_df['index_id'] = [i for i in range(len(act_df))]

	print(act_df)
	# user_id, history, indices, values
	print(pred_df)
	# user_id, rec_items, rec_predictions
	index_id_list = act_df['index_id'].tolist()

	act_df = act_df[act_df['values']>0]
	act_df = act_df.sample(frac = 1)
	categories, bins = pd.qcut(act_df['values'], q=10, retbins=True,labels = range(1,11))
	act_df['act_level'] = categories.astype(int)
	print(bins)
# 	[0.00379577 0.03991271 0.04975198 0.05732215 0.06494929 0.0738075
#  0.08456758 0.0982686  0.11746017 0.15157312 0.76564592]
	# [1.38556957e-03 1.13839866e+00 1.46889286e+00 1.72234608e+00
	# 1.98681581e+00 2.29951990e+00 2.67184877e+00 3.14932599e+00
	# 3.84628472e+00 5.20543056e+00 1.76705456e+01]

	act_df.rename({"indices":'latent_id'},axis = 1,inplace = True)
	act_df.sort_values(['latent_id','values'],ascending=[True,False],inplace=True)
	
	latent_info = act_df.groupby('latent_id').agg({'values':['max','mean','min','count'],'act_level':['max','mean','min','count']})
	print(latent_info.describe(percentiles=[i/10 for i in range(1,10)]))

	latent_dict = dict(list(act_df.groupby('latent_id')))
	print('num of latents',len(latent_dict))
	
	SELECT_NUM = 5
	final_selection_dict = {}
	for latent_id in latent_dict:
		df = latent_dict[latent_id]
		high_df = df[df['act_level']>=5]
		print('[latent id]',latent_id,len(df),len(df[df['act_level']>=5]))
		if len(high_df) == 0:
			continue
		high_activation_list = high_df.head(10)['index_id'].tolist()
		high_act_value_list = high_df.head(10)['act_level'].tolist()

		latent_result = {}
		all_numbers = list(range(len(high_activation_list)))
		if len(high_activation_list)<=SELECT_NUM:
			# latent_result['train'] = high_activation_list
			# latent_result['train_acts'] = high_act_value_list
			selected_numbers = all_numbers
			unselected_numbers =  all_numbers #random.sample(all_numbers, 1)
		else:
			selected_numbers = random.sample(all_numbers, SELECT_NUM)
			unselected_numbers = [num for num in all_numbers if num not in selected_numbers]
			if len(unselected_numbers) < SELECT_NUM:
				numbers = random.sample(selected_numbers, SELECT_NUM - len(unselected_numbers))
				unselected_numbers = unselected_numbers + numbers

		selection = {'train':selected_numbers,'test':unselected_numbers}
		for key in ['train','test']:
			latent_result[key] = [high_activation_list[index] for index in selection[key]]
			latent_result[f"{key}_acts"] = [high_act_value_list[index] for index in selection[key]]

		selected_index_ids = df['index_id'].tolist()
		remaining_users = [index_id for index_id in index_id_list if index_id not in selected_index_ids]
		random_selection = random.sample(remaining_users, SELECT_NUM)
		latent_result['random'] = random_selection
		final_selection_dict[latent_id] = latent_result
		# import ipdb;ipdb.set_trace()
	
	print(len(latent_dict),len(final_selection_dict))
	result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__trainTest_info_norm_haveTestAll{method}.jsonl"
	with open(result_file,'w') as f:
		for key, value in final_selection_dict.items():
			json.dump({key: value}, f)
			f.write("\n")

	result_file = f"../../log/{base_model}_SAE/result_file/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__prediction_norm_haveTestAll{method}.csv"
	pred_df.to_csv(result_file,sep='\t',index=False)
	




