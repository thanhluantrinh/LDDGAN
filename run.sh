#!/bin/sh
export MASTER_PORT=6038
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- Wavelet -----------
if [[ $MODE == train ]]; then
	echo "==> Training"
	if [[ $DATASET == cifar10 ]]; then
		python3 train_ldgan.py --dataset cifar10 --exp atn32_g122_2block_d3_Recloss_nz50_SmL --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 4000 --ngf 64 --nz 50 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 --save_content --datadir ./data/cifar-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS --save_ckpt_every 25 \
			--current_resolution 16 --attn_resolutions 32 --num_disc_layers 3  --scale_factor 105.0 \
			--no_lr_decay \
            --AutoEncoder_config autoencoder/config/cifar10_16x16x4.yaml \
            --AutoEncoder_ckpt autoencoder/weight/16x16x4_551.ckpt \
			--rec_loss \
			--sigmoid_learning \

	elif [[ $DATASET == celeba_256_reaD ]]; then
		python3 train_ldgan_celeba_reaD.py --dataset celeba_256 --image_size 256 --exp g1222_128_2block_d4_attn16_2step_reaD --num_channels 3 --num_channels_dae 128 --ch_mult 1 2 2 2 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 2. \
			--nz 100 --z_emb_dim 256 --lr_d 1.0e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 4 --rec_loss \
			--save_content_every 1 \
            --AutoEncoder_config ./autoencoder/config/CELEBA_config.yaml \
            --AutoEncoder_ckpt ./autoencoder/weight/CELEBA_weight.ckpt \
			--scale_factor 6.0 \

	elif [[ $DATASET == lsun ]]; then
		python3 train_ldgan.py --dataset lsun --image_size 256 --exp g11222_128_3block_d4_attn32_nz50 --num_channels 4 --num_channels_dae 128 --ch_mult 1 1 2 2 2 --num_timesteps 4 \
			--num_res_blocks 3 --batch_size 16 --num_epoch 1000 --ngf 64 --embedding_type positional --ema_decay 0.9999 --r1_gamma 1.0 \
			--nz 50 --z_emb_dim 256 --lr_d 1.2e-4 --lr_g 1.6e-4 --lazy_reg 10 --save_content --datadir data/lsun/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolution 32 --num_disc_layers 4  \
			--save_content_every 1 \
            --AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
            --AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--scale_factor 60.0 \
			--no_lr_decay \
			--rec_loss --sigmoid_learning \
			#--not_use_tanh
			#--use_ema --rec_loss


	fi
else
	echo "==> Testing Latent DDGAN"
	#TODO: need to set tanhout in Decoder
	if [[ $DATASET == cifar10 ]]; then \
		python3 test_ldgan_old.py --dataset cifar10 --exp best --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 50 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 --epoch_id 3800 \
			--image_size 32 --current_resolution 16 --attn_resolutions 32 \
            --scale_factor 105.0 \
            --AutoEncoder_config autoencoder/config/cifar10_16x16x4.yaml \
            --AutoEncoder_ckpt autoencoder/weight/16x16x4_551.ckpt \
			--batch_size 100 --measure_time \
			#--compute_fid --real_img_dir pytorch_fid/cifar10_train_stat.npy \
			#Recloss 1400
			#--batch_size 100 --measure_time \
			#Recloss--epoch_id 3625 -> FID 3.615
			#NoRecloss--epoch_id 3350 -> FID 3.77
			#Recloss, SmL--epoch 3800 -> FID 3.348

	elif [[ $DATASET == lsun_loop ]]; then \
		python3 test_ldgan_UViT_loop.py --dataset lsun --exp g1222_128_2block_d4_attn16_nz100_notanh --num_channels 4 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 2500 \
			--image_size 256 --current_resolution 32 --attn_resolutions 16 \
            --scale_factor 1.0 \
            --AutoEncoder_config ./autoencoder/config/LSUN_config.yaml \
            --AutoEncoder_ckpt ./autoencoder/weight/LSUN_weight.ckpt \
			--batch_size 100 --compute_fid --real_img_dir pytorch_fid/lsun_church_stat.npy \
			--not_use_tanh \


	elif [[ $DATASET == stl10 ]]; then
		python test_wddgan.py --dataset stl10 --exp wddgan_stl10_exp1_atn16_wg1222_d4_recloss_900ep --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 800 \
			--image_size 64 --current_resolution 32 --attn_resolutions 16 \
			--net_type wavelet \
			--use_pytorch_wavelet \
			# --compute_fid --real_img_dir pytorch_fid/stl10_stat.npy \
			# --batch_size 100 --measure_time \

	elif [[ $DATASET == celeba_256 ]]; then
		python test_ldgan_celeba.py --dataset celeba_256 --image_size 256 --exp g1222_128_2block_d4_attn16_2step_reaD --num_channels 3 --num_channels_dae 128 \
			--ch_mult 1 2 2 2 --num_timesteps 2 --num_res_blocks 2  --epoch_id 200 \
			--current_resolution 64 --attn_resolutions 16 \
            --AutoEncoder_config ./autoencoder/config/CELEBA_config.yaml \
            --AutoEncoder_ckpt ./autoencoder/weight/CELEBA_weight.ckpt \
			--scale_factor 6.0 \
			--batch_size 100 --measure_time \
			#--compute_fid --real_img_dir ./pytorch_fid/celebahq_stat.npy \
			#--batch_size 100
	fi
fi
