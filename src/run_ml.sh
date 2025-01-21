# python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --gpu 1


python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 16 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 8 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1



python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 8 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 16 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1


python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 8 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 16 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 4 --sae_k 8 --sae_scale_size 16 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
