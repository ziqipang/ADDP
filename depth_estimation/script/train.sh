export BASE_DATA_DIR=/work/nvme/bbsg/xuxalan/datasets  # directory of training data
export BASE_CKPT_DIR=checkpoints  # directory of pretrained checkpoint

CUDA_VISIBLE_DEVICES=1 python train.py --config config/train_marigold.yaml --output_dir delta_logs/blur_aug_int20 --do_not_copy_data
