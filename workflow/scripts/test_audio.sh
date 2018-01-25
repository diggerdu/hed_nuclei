#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
expName=naive
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python test.py \
 --serial_batches \
 --Path "/home/alan/data/fucking" \
 --nClasses 12\
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --nThreads 6 \
 --nfft 256 --hop 128 --nFrames 128 --batchSize  5\
 --split_hop 0 \
 --niter 1 --niter_decay 30 \
 --lr 1e-6 \
 --gpu_ids 0 \
