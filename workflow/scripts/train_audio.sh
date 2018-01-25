#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
expName=naive
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --Path "/home/diggerdu/dataset/tfsrc/train/audio/" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --nThreads 6 \
 --nfft 256 --hop 128 --nFrames 128 --batchSize  26\
 --split_hop 0 \
 --niter 100000000000000000000000000000000000 --niter_decay 30 \
 --lr 1e-4 \
 --gpu_ids 0 \
#  --serial_batches
