model="hicmae_pretrain_base"
OUTPUT_DIR="./saved/model/pretraining/voxceleb2/audio_visual/${model}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
# Set the path to pre-training dataset.
DATA_PATH='./saved/data/voxceleb2/info_clean_av_new.csv'
# batch_size can be adjusted according to number of GPUs
# this script is for 4 GPUs (1 nodes x 4 GPUs)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 11120 \
        run_mae_pretraining_av.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --input_size 160 \
        --mask_ratio_audio 0.8 \
        --input_size_audio 256 \
        --model pretrain_hicmae_dim512_patch16_160_a256 \
        --encoder_depth 10 \
        --decoder_depth 4 \
        --encoder_depth_audio 10 \
        --decoder_depth_audio 4 \
        --encoder_fusion_depth 2 \
        --batch_size 40 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 16 \
        --roll_mag_aug True \
        --return_intermediate_features 3 6 9 \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07 \
        --use_frame_diff_as_target \
        >${OUTPUT_DIR}/nohup.out 2>&1 &

