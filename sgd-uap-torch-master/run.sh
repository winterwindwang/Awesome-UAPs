#!/bin/bash
# Fixed Params
PRETRAINED_DATASET="imagenet"
DATASET="imagenet"
EPSILON=0.03922
BATCH_SIZE=50
NGPU=1
# nohup bash -u run.sh > train_uaps_revised_number_images_2000_230619.out 2>&1 &
TARGET_NETS="resnet101 resnet152"
# TARGET_NETS="vgg16 vgg19 resnet50 resnet101 resnet152 resnext50_32x4d wide_resnet50_2 efficientnet_b0 densenet121 densenet161 alexnet googlenet mnasnet1_0"
# TARGET_NETS="vgg16 vgg19 resnet50 resnet101 resnet152 resnext50_32x4d wide_resnet50_2 efficientnet_b0 densenet121 densenet161 googlenet mnasnet1_0"
for target_net in $TARGET_NETS; do
    python3 train_uap.py \
      --dataset $DATASET \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --epsilon $EPSILON \
      --batch_size $BATCH_SIZE \
      --ngpu $NGPU
done
