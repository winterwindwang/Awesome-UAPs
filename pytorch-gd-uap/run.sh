TARGET_NETS="vgg16 vgg19 resnet50 resnet101 resnet152 resnext50 wideresnet efficientnetb0 densenet121 densenet161 googlenet mnasnet10"
for target_net in $TARGET_NETS; do
    python3 train.py \
      --model $target_net
done
