import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

if __name__=='__main__':

	# version = "v5_baseline"
	version = "v5"
	# version = "v5_recModel"
	
	# base_model = "SASRec"
	# base_model = "BPRMF"
	# base_model = "TiMiRec"
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
	
	lr_dict = {"SASRec":SASRec_lr_dict,'BPRMF':bprmf_lr_dict,"LightGCN":LightGCN_lr_dict,"TiMiRec":TiMiRec_lr_dict}
	batch_dict = {"SASRec":SASRec_batch_dict,"BPRMF":bprmf_batch_dict,"LightGCN":LightGCN_batch_dict,'TiMiRec':TiMiRec_batch_dict}
	
	sae_lr = lr_dict[base_model][dataset_name][sae_k]
	batch_size = batch_dict[base_model][dataset_name][sae_k]



	# latent_dict_file = f"../../log/SASRec_SAE/result_file/latents/SASRec_SAE__Grocery_and_Gourmet_Food__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}_latentDict_v31.jsonl"
	# with open(latent_dict_file, 'r') as file:
	# 	result_dict = json.load(file.readline())# dataset_name = "ML_1MTOPK"
	# sae_lr = "1e-05"
	# batch_size = 16
	# latent_dict_file = f"../../log/SASRec_SAE/result_file/latents/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}_latentDict_verification_v31.jsonl"
	latent_dict_file = f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_norm_allInfo_{version}.jsonl"
	with open(latent_dict_file, 'r') as file:
		latent_info = json.loads(file.readline())

	info_dict = dict()
	for latent_id in latent_info:
		print('[latent id]',latent_id)
		content = latent_info[latent_id]
		# pred_test, pred_rand
		for key in ['pred_test','pred_rand']:
			if key not in content:
				print(f"[pred_test] is none Latent={latent_id}")
				continue
			try:
				ans = [float(s) for s in content[key]]
				ans = [s if 0<=s<=10 else -1 for s in ans]
				content[f"{key}_value"] = ans
			except ValueError:
				raise ValueError(f"Input '{content[key]}' is not a valid integer or is out of range 0-10.")
		
		# pred_test_value, test_acts
		recall = 0
		length = 0
		if 'pred_test_value' in content:
			for value in content['pred_test_value']:
				if value > 5:
					recall += 1
			length = len(content['pred_test_value'])
		precision = 0
		for value in content['pred_rand_value']:
			if value <=5:
				precision += 1
		content['confidence_score'] = (precision + recall) / (length + len(content['pred_rand_value']) )
		if length != 0:
			content['recall'] = recall / length
		else:
			content['recall'] = 0
		content['precision'] = precision / len(content['pred_rand_value'])
		info_dict[latent_id] = content
		print(f"{content['confidence_score']:4f}\t{content['recall']:.4f}\t{content['precision']:.4f}")

	df_info = pd.DataFrame(index = list(info_dict.keys()))
	df_info['latent_id'] = df_info.index
	for key in ['recall','precision','confidence_score']:
		df_info[key] = df_info['latent_id'].apply(lambda v:info_dict[v][key])
	print(df_info.describe(percentiles=[i/10 for i in range(1,10)]))


	total_count = len(df_info)
	bin_edges = np.arange(-0.05, 1.06, 0.1)
	bin_edges_v0 = np.arange(-0.1, 1.1, 0.2)

	# Plotting the distributions for recall, precision, and confidence_score as percentages
	# plt.title("MovieLens 1M Dataset",fontsize = 18)
	plt.figure(figsize=(12, 4))
	gs = gridspec.GridSpec(1, 3, width_ratios=[2,2,3])

	# Recall distribution as percentage
	# plt.subplot(1, 3, 1)
	plt.subplot(gs[0])
	plt.hist(df_info['recall'], bins=bin_edges_v0, edgecolor='black', color = 'green',weights=np.ones(total_count) / total_count * 100)
	plt.title('Recall Distribution',fontsize=16)
	plt.xlabel('Recall',fontsize=12)
	plt.ylabel('Percentage (%)')
	plt.xticks(np.arange(0, 1.1, 0.2))
	plt.ylim(0, 80)  # Setting a common y-axis limit for consistency

	# Precision distribution as percentage
	# plt.subplot(1, 3, 2)
	plt.subplot(gs[1])
	plt.hist(df_info['precision'], bins=bin_edges_v0, edgecolor='black',color='green', weights=np.ones(total_count) / total_count * 100)
	plt.title('Precision Distribution',fontsize=16)
	plt.xlabel('Precision',fontsize=12)
	plt.ylabel('Percentage (%)')
	plt.xticks(np.arange(0, 1.1, 0.2))
	plt.ylim(0, 80)  # Setting the same y-axis limit

	# Confidence score distribution as percentage
	# plt.subplot(1, 3, 3)
	plt.subplot(gs[2])
	plt.hist(df_info['confidence_score'], bins=bin_edges, edgecolor='black',color = 'green', weights=np.ones(total_count) / total_count * 100)
	plt.title('Confidence Score Distribution',fontsize=16)
	plt.xlabel('Confidence Score',fontsize=12)
	plt.ylabel('Percentage (%)')
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.ylim(0, 30)  # Consistent y-axis limit

	title_name = {"ML_1MTOPK":"MovieLens 1M",\
			   "Grocery_and_Gourmet_Food":"Amazon Grocery",\
				'LastFM':"LastFM",\
				'LastFM_8k':"LastFM 8k"}
	plt.suptitle(f"{title_name[dataset_name]} (k={sae_k})", fontsize=18, y=0.96)
	plt.tight_layout()
	plt.savefig(f"../../log/{base_model}_SAE/figs/0_{dataset_name}_({sae_k},{sae_scale_size})_{batch_size}_{sae_lr}_allInfo_{version}.png")

	print(df_info [df_info['confidence_score']>0.8])

	print(len(df_info [df_info['confidence_score']==1.0]),len(df_info [df_info['confidence_score']>=0.9]),len(df_info [df_info['confidence_score']>0.8]),len(df_info [df_info['confidence_score']>=0.8]),len(df_info))
	
	result_file = f"../../log/{base_model}_SAE/result_file/latents/{base_model}_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_batch_size={batch_size}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__latentDict_verification_result_{version}.csv"
	df_info.to_csv(result_file)
	# # import ipdb;ipdb.set_trace()


	

	# 2026 all latents
	# 0.8 - 398
	# 0.9 - 168


# 	            recall    precision  confidence_score
# count  2026.000000  2026.000000       2026.000000
# mean      0.543863     0.641461          0.593422
# std       0.299994     0.257737          0.179482
# min       0.000000     0.000000          0.000000
# 10%       0.200000     0.200000          0.400000
# 20%       0.200000     0.400000          0.400000
# 30%       0.400000     0.600000          0.500000
# 40%       0.400000     0.600000          0.500000
# 50%       0.600000     0.600000          0.600000
# 60%       0.600000     0.800000          0.600000
# 70%       0.800000     0.800000          0.700000
# 80%       0.800000     0.800000          0.700000
# 90%       1.000000     1.000000          0.800000
# max       1.000000     1.000000          1.000000


#      latent_id  recall  precision  confidence_score
# 11          11     1.0        1.0               1.0
# 22          22     0.8        1.0               0.9
# 57          57     1.0        0.8               0.9
# 59          59     0.8        1.0               0.9
# 60          60     1.0        0.8               0.9
# ...        ...     ...        ...               ...
# 1972      1972     1.0        0.8               0.9
# 1993      1993     1.0        1.0               1.0
# 1999      1999     1.0        1.0               1.0
# 2025      2025     1.0        0.8               0.9
# 2040      2040     1.0        0.8               0.9



#      latent_id  recall  precision  confidence_score
# 9            9     1.0        0.8          0.900000
# 72          72     0.0        1.0          0.833333
# 401        401     1.0        0.6          0.800000
# 414        414     0.0        1.0          0.833333
# 607        607     1.0        0.8          0.900000
# 614        614     1.0        0.8          0.900000
# 733        733     1.0        0.6          0.800000
# 763        763     0.8        0.8          0.800000
# 947        947     1.0        0.8          0.900000
# 1160      1160     1.0        0.8          0.833333
# 1396      1396     1.0        0.6          0.800000
# 1477      1477     0.8        0.8          0.800000
# 1544      1544     0.8        0.8          0.800000
# 1548      1548     0.0        1.0          0.833333
# 1562      1562     0.0        1.0          0.833333
# 1726      1726     1.0        0.6          0.800000

#             recall    precision  confidence_score
# count  1946.000000  1946.000000       1946.000000
# mean      0.732451     0.114902          0.407239
# std       0.401321     0.174208          0.200672
# min       0.000000     0.000000          0.000000
# 10%       0.000000     0.000000          0.000000
# 20%       0.000000     0.000000          0.166667
# 30%       0.800000     0.000000          0.375000
# 40%       0.800000     0.000000          0.428571
# 50%       1.000000     0.000000          0.500000
# 60%       1.000000     0.000000          0.500000
# 70%       1.000000     0.200000          0.500000
# 80%       1.000000     0.200000          0.555556
# 90%       1.000000     0.400000          0.600000
# max       1.000000     1.000000          0.900000