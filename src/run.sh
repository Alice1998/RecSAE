# train the SASRec or load your own model
# For other model, see ./docs/demo_scripts_results
python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1

# train the RecSAE plug-in
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

# inference the RecSAE
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --s


cd analysis
python 1_pack_trainResult.py
python 2_construct_latentDict.py
python 3_analysis_verResult.py