# RecSAE

RecSAE: Recommendation Model Interpretation with Sparse Autoencoder

We use the [ReChorus](https://github.com/THUwangcy/ReChorus) framework as our code base and implement the SAE module upon it.


### Command

```bash

# data preparation
# ./data/Grocery_and_Gourmet_Food/Amazon.ipynb
# ./data/MovieLens_1M/MovieLens-1M.ipynb


cd src

# 1.0 train the recommendation model or load a trained model.
python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1

# 1.1 train the RecSAE module.
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

# 1.2 inference the RecSAE
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 0

# Other Model
# BPRMF
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1
# TiMiRec
python main.py --model_name TiMiRec --dataset Grocery_and_Gourmet_Food --path '../data' --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --K 6 --add_pos 1 --add_trm 1 --stage pretrain --test_all 1
python main.py --model_name TiMiRec --dataset Grocery_and_Gourmet_Food --path '../data' --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --K 6 --add_pos 1 --add_trm 1 --stage finetune --temp 1 --n_layers 1 --test_all 1

# Dataset
# Amazon: --dataset 'Grocery_and_Gourmet_Food' --path '../data'
# MovieLens: --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M'
# LastFM: --dataset 'LastFM' --path '../data'


# 2 construct concept descriptions and confidence score
cd analysis
python 1_pack_trainResult.py
python 2_construct_latentDict.py
python 3_analysis_verResult.py
```

### Hyper-parameter
see ./src/analysis/1_pack_trainResult.py


