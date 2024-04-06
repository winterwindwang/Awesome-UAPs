#!/bin/bash

# Fixed Params
PRETRAINED_DATASET="imagenet"
DATASET="imagenet"
EPSILON=0.03922
LOSS_FN="bounded_logit_neg"
TARGET_CLASS=150
CONFIDENCE=10
BATCH_SIZE=25
LEARNING_RATE=0.005
NUM_ITERATIONS=1000
NB_IMAGES=2000
WORKERS=4
NGPU=1
SUBF="imagenet_new"

# nohup bash -u run.sh > train_uaps_df_uap_number_images_2000_bounded_logit_neg_230618.out 2>&1 &  
# LOSS_FN="bounded_logit_fixed_ref"  bounded_logit_neg, bounded_logit, bounded_logit_neg  bounded_logit_neg
# TARGET_NETS="vgg16 vgg19 resnet50 resnet152 densenet121 densenet161 alexnet"
TARGET_NETS="vgg16 vgg19 resnet50 resnet101 resnet152 resnext50 wideresnet efficientnetb0 densenet121 densenet161 googlenet mnasnet10"
# --target_class $TARGET_CLASS --targeted \

for target_net in $TARGET_NETS; do
    python3 /mnt/jfs/wangdonghua/pythonpro/uap_virtual_data.pytorch-master/train_uap.py \
      --dataset $DATASET \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --epsilon $EPSILON \
      --loss_function $LOSS_FN --confidence $CONFIDENCE \
      --num_iterations $NUM_ITERATIONS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF \
      --nb_images $NB_IMAGES
done
