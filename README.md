# Image Classification - ImageNet

In this task, the model learns to predict the class of an image, out of 1000 classes.

## Requirements

- python >= 3.7
- python libraries:
```bash
pip install -r requirements.txt
```

## Data preparation

We use the standard ImageNet dataset, you can download it from http://image-net.org/. Validation images are put in labeled sub-folders. The file structure should look like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Training
To train a learngene on `DeiT-Tiny` for 100epochs, run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py  --data-path /home/xxxx/datasets/ImageNet2012/Data/CLS-LOC --cfg configs/deit_tiny_hard_gfish_half_qkv+mlp_gaussian.yaml
```

## Fine-tuning
To Fine-tune for downstream datasets (e.g., ImageNet, CIFAR10, CIFAR100...), run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346  main.py  --cfg configs/deit_tiny_hard_gfish_half_qkv+mlp_gaussian_downstream.yaml --finetune ./output/deit_learngene_tiny/4gpu_lr=5e-4_warm=5_attn2to3_ffn6to12_initialize_100epochs/ckpt_best.pth
```
